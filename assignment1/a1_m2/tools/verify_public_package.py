"""Verify a publication ZIP and optionally smoke-test its fresh extraction.

The archive, rather than the research checkout, is the input to this release
gate. Software checks use only the extracted source and explicitly supplied
third-party dependency directories. They run synthetic tests, never a measured
training study, real-image inference or an official-test assessment.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
import zipfile


MANIFEST = "PUBLICATION_MANIFEST.json"
REQUIRED_PATHS = (
    "garmentlab/prospective.py",
    "notebooks/A1_Optimize_A100_v3_1_7Days.ipynb",
    "notebooks/A1_Optimize_A100_v3_1_Fix.ipynb",
    "notebooks/A1_Optimize_A100_v3.ipynb",
)
FROZEN_SOURCE = "assets/frozen_sources/CO3133_Confirmatory_Source_20261029.zip"
FROZEN_SOURCE_SHA256 = "a1d7a73a01ef749081516d432bbb3f839c134c70ee4fa0b655a87948ccc514bf"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_relative(name):
    require(isinstance(name, str) and name and "\\" not in name,
            f"Unsafe archive path: {name!r}")
    parts = name.split("/")
    if parts[-1] == "":
        parts = parts[:-1]
    require(parts and all(p not in ("", ".", "..") and ":" not in p for p in parts)
            and not PurePosixPath(name).is_absolute(), f"Unsafe archive path: {name!r}")
    return PurePosixPath(*parts)


def digest_stream(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def inspect_integrity(archive, required_paths=REQUIRED_PATHS):
    """Dependency-free inventory gate, shared with the regression tests."""
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        require(len(names) == len(set(names)), "Duplicate archive members are forbidden")
        require(len(names) == len({n.casefold() for n in names}),
                "Case-colliding archive members are forbidden")
        for info in infos:
            safe_relative(info.filename)
            require(not stat.S_ISLNK(info.external_attr >> 16),
                    f"Archive symlink is forbidden: {info.filename}")
        files = {i.filename: i for i in infos if not i.is_dir()}
        require(MANIFEST in files, f"Missing {MANIFEST}")
        manifest = json.loads(z.read(MANIFEST))
        entries = manifest["payloads"]
        require(isinstance(entries, list), "Manifest payloads must be a list")
        indexed = {item["path"]: item for item in entries}
        require(len(indexed) == len(entries), "Duplicate manifest paths are forbidden")
        for name in indexed:
            safe_relative(name)
        require(set(files) == set(indexed) | {MANIFEST},
                "Manifest and archive file inventories differ")
        missing = sorted(set(required_paths) - set(files))
        require(not missing, "Missing required publication files: " + ", ".join(missing))
        for name, item in indexed.items():
            require(files[name].file_size == item["bytes"], f"Payload size differs: {name}")
            with z.open(name) as stream:
                require(digest_stream(stream) == item["sha256"], f"Payload hash differs: {name}")
        require(z.testzip() is None, "ZIP CRC verification failed")
    with Path(archive).open("rb") as stream:
        archive_sha = digest_stream(stream)
    return {"status": "PASS_ARCHIVE_INVENTORY_AND_HASHES", "archive_sha256": archive_sha,
            "payloads_verified": len(indexed), "required_paths_verified": list(required_paths)}


def notebook_code(data, name):
    notebook = json.loads(data)
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    require(cells, f"Notebook has no code cells: {name}")
    require(all(not c.get("outputs") and c.get("execution_count") is None for c in cells),
            f"Notebook retains execution outputs/counts: {name}")
    source = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
              for c in cells]
    # The historical optimization notebooks and the confirmation notebook use
    # Python cells. The separate M1 notebook legitimately contains IPython magic.
    return [ast.dump(ast.parse(s)) for s in source]


def inspect_study_reference(z):
    """Verify the packaged upgrade profile's historical dependency closure."""
    names = set(z.namelist())
    profile_name = "configs/a1_upgrade_study.json"
    require(profile_name in names, "Missing declared upgrade-study profile: " + profile_name)
    profile = json.loads(z.read(profile_name))
    reference = safe_relative(profile["reference_run"]).as_posix()
    prefix = reference + "/"
    require(any(n.startswith(prefix) for n in names),
            "Missing historical study reference directory: " + reference)
    freeze_name = prefix + "freeze_manifest.json"
    export_name = prefix + "artifact_manifest.json"
    for name in (freeze_name, export_name):
        require(name in names, "Missing historical study reference manifest: " + name)
    freeze_bytes, export_bytes = z.read(freeze_name), z.read(export_name)
    require(hashlib.sha256(freeze_bytes).hexdigest() == profile["reference_freeze_sha256"],
            "Historical reference freeze hash differs from upgrade profile")
    require(hashlib.sha256(export_bytes).hexdigest() == profile["reference_artifact_manifest_sha256"],
            "Historical reference artifact-manifest hash differs from upgrade profile")
    freeze, exported = json.loads(freeze_bytes), json.loads(export_bytes)
    body = {key: value for key, value in freeze.items() if key != "manifest_sha256"}
    body_hash = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False).encode("utf-8")).hexdigest()
    require(body_hash == freeze["manifest_sha256"], "Historical reference freeze self-hash differs")
    for freeze_key, profile_key in (("run_id", "reference_run_id"),
                                    ("split_sha256", "expected_split_sha256"),
                                    ("dataset_sha256", "expected_dataset_sha256")):
        require(freeze[freeze_key] == profile[profile_key],
                "Historical reference identity differs: " + freeze_key)
    required = dict(freeze["files_sha256"])
    for relative, expected in exported.items():
        require(relative not in required or required[relative] == expected,
                "Historical freeze and export disagree: " + relative)
        required[relative] = expected
    for relative, expected in required.items():
        member = prefix + safe_relative(relative).as_posix()
        require(member in names, "Historical reference payload is missing: " + member)
        with z.open(member) as stream:
            require(digest_stream(stream) == expected, "Historical reference payload hash differs: " + member)
    base_name = safe_relative(profile["base_config"]).as_posix()
    require(base_name in names, "Upgrade profile base configuration is missing: " + base_name)
    current = json.loads(z.read(base_name))
    recorded = json.loads(z.read(prefix + "config.json"))
    current["device"] = recorded["device"]
    require(current == recorded, "Upgrade profile base recipe differs from its historical reference")
    return {"profile": profile_name, "reference_run": reference,
            "freeze_locked_payloads_verified": len(freeze["files_sha256"]),
            "artifact_manifest_payloads_verified": len(exported),
            "unique_historical_payloads_verified": len(required)}


def root_path_parts(node):
    if isinstance(node, ast.Name) and node.id == "ROOT":
        return []
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        previous = root_path_parts(node.left)
        if previous is not None and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
            return previous + node.right.value.replace("\\", "/").split("/")
    return None


def inspect_contract(archive, reference_notebooks=None):
    """Check source identity and notebook fixtures without importing project code."""
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        require(FROZEN_SOURCE in names, f"Missing immutable source archive: {FROZEN_SOURCE}")
        frozen_data = z.read(FROZEN_SOURCE)
        require(hashlib.sha256(frozen_data).hexdigest() == FROZEN_SOURCE_SHA256,
                "Immutable confirmation source archive differs")
        with zipfile.ZipFile(io.BytesIO(frozen_data)) as frozen:
            modules = [n for n in frozen.namelist() if n.endswith(".py")
                       and n.startswith(("a1/", "a1opt/", "a1confirm/", "garmentlab/"))]
            require(modules, "Immutable source archive has no model/runner modules")
            for name in modules:
                require(name in names, f"Frozen source module missing from main source: {name}")
                require(z.read(name) == frozen.read(name), f"Frozen main-source module differs: {name}")
        fixtures = set()
        for name in sorted(n for n in names if n.startswith("tests/") and n.endswith(".py")):
            tree = ast.parse(z.read(name).decode("utf-8-sig"), filename=name)
            for node in ast.walk(tree):
                parts = root_path_parts(node)
                if parts and parts[0] == "notebooks" and parts[-1].endswith(".ipynb"):
                    fixture = "/".join(parts)
                    require(fixture in names,
                            f"Packaged test {name} references a missing notebook: {fixture}")
                    fixtures.add(fixture)
        parsed = {}
        for name in sorted(fixtures | set(REQUIRED_PATHS[1:])):
            parsed[name] = notebook_code(z.read(name), name)
        canonical = "notebooks/A1_Optimization_A100.ipynb"
        seven_days = REQUIRED_PATHS[1]
        if canonical in names:
            require(notebook_code(z.read(canonical), canonical) == parsed[seven_days],
                    "Published OPT3 notebook code differs from its seven-day regression fixture")
        references_checked = []
        if reference_notebooks is not None:
            reference_notebooks = Path(reference_notebooks)
            for name in REQUIRED_PATHS[1:]:
                original = reference_notebooks / Path(name).name
                require(original.is_file(), f"Reference notebook is missing: {original}")
                require(notebook_code(original.read_bytes(), name) == parsed[name],
                        f"Packaged historical notebook code differs from reference: {name}")
                references_checked.append(name)
        historical_reference = inspect_study_reference(z)
    return {"frozen_main_source_modules_verified": len(modules),
            "static_notebook_fixtures_verified": sorted(fixtures),
            "reference_notebook_code_AST_verified": references_checked,
            "historical_upgrade_reference_verified": historical_reference,
            "scope": "Notebook syntax/output policy and explicit ROOT/notebooks fixture paths are checked. Dynamic file dependencies are exercised by the software phase, not inferred exhaustively."}


def extract_fresh(archive, destination):
    destination = Path(destination).resolve()
    require(not destination.exists(), "Extraction directory must not exist: " + str(destination))
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            relative = safe_relative(info.filename)
            target = destination.joinpath(*relative.parts)
            require(target.resolve().is_relative_to(destination), "Extraction path escaped its root")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, target.open("xb") as dst:
                    for block in iter(lambda: src.read(1024 * 1024), b""):
                        dst.write(block)
    return destination


def software_checks(root, dependency_paths=(), timeout=300):
    root = Path(root).resolve()
    dependencies = []
    for value in dependency_paths:
        value = Path(value).resolve()
        require(value.is_dir(), f"Dependency directory does not exist: {value}")
        require(not any((value / package).exists() for package in ("a1", "a1opt", "a1confirm", "garmentlab")),
                "Dependency paths cannot provide project source: " + str(value))
        dependencies.append(value)
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join(map(str, [root, *dependencies])),
               PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    import_check = ("from pathlib import Path; import garmentlab.prospective as p; "
                    "expected=Path('garmentlab/prospective.py').resolve(); "
                    "assert Path(p.__file__).resolve()==expected,(p.__file__,str(expected)); "
                    "print('PASS_EXTRACTED_PROSPECTIVE_IMPORT')")
    commands = [
        ("main_source_import", [sys.executable, "-c", import_check]),
        ("confirmatory_cli_help", [sys.executable, "tools/run_confirmatory.py", "--help"]),
        ("historical_7days_notebook", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_a1opt_notebook_7days.py"]),
        ("historical_v31_notebook", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_a1opt_notebook_v31.py"]),
        ("historical_upgrade_study", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_a1_study.py"]),
        ("complete_cpu_software_suite", [sys.executable, "-m", "pytest", "tests", "-q", "--basetemp", str(root.parent / (root.name + "_pytest")), "--junitxml", str(root.parent / (root.name + "_junit.xml"))]),
    ]
    records = []
    for label, command in commands:
        result = subprocess.run(command, cwd=root, env=env, text=True, encoding="utf-8",
                                errors="replace", capture_output=True, timeout=timeout)
        records.append({"name": label, "command": command[1:], "returncode": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr})
        require(result.returncode == 0, f"Extracted-package software check failed: {label}\n{result.stdout}\n{result.stderr}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, help="New extraction directory; must not already exist")
    parser.add_argument("--software", action="store_true", help="Run CPU-only imports, CLI and synthetic regression tests")
    parser.add_argument("--dependency-path", type=Path, action="append", default=[], help="Additional third-party package directory, never a project checkout")
    parser.add_argument("--reference-notebooks", type=Path, help="Optional directory containing the three original historical notebooks")
    parser.add_argument("--receipt", type=Path, help="Write the verification JSON outside the extracted publication")
    args = parser.parse_args()
    receipt = {"status": "FAIL_PUBLIC_PACKAGE_GATE"}
    try:
        receipt.update(inspect_integrity(args.archive))
        receipt.update(inspect_contract(args.archive, args.reference_notebooks))
        if args.work_dir or args.software:
            destination = args.work_dir or Path(tempfile.mkdtemp(prefix="co3133-package-gate-")) / "checkout"
            extracted = extract_fresh(args.archive, destination)
            receipt["extraction_directory"] = str(extracted)
            if args.software:
                receipt["software_checks"] = software_checks(extracted, args.dependency_path)
        receipt["status"] = "PASS_PUBLIC_PACKAGE_GATE"
        receipt["software_phase_executed"] = bool(args.software)
    except (AssertionError, KeyError, OSError, ValueError, TypeError, SyntaxError,
            zipfile.BadZipFile, subprocess.TimeoutExpired) as error:
        receipt["status"] = "FAIL_PUBLIC_PACKAGE_GATE"
        receipt["error"] = str(error)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "PASS_PUBLIC_PACKAGE_GATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
