"""Build the local GitHub Pages study and Markdown from verified publication data.

No hosting, registration, push, training or inference. Generated project pages
and the GarmentLab result section are written from recorded data. Existing
M1/A2/A3 pages and measured inputs stay intact.
"""
from __future__ import annotations

import argparse
import html
import json
import statistics
import struct
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output/a1_final/data/publication_data.json"
REPO = "https://github.com/cau26/CO3133-DLProject"
NAMES = {"linear": "Linear", "mlp": "MLP", "cnn": "CNN", "gru": "GRU", "transformer": "Transformer"}
def preferred_artifact(latest, original):
    return latest if (ROOT / latest).is_file() else original


REPORT = preferred_artifact("output/pdf/G-M10_A1_Report_Polished_20261028.pdf", preferred_artifact("output/pdf/G-M10_A1_Report_Improve_20261028.pdf", "output/pdf/G-M10_A1_Report_Final.pdf"))
TEX = preferred_artifact("output/latex/G-M10_A1_Overleaf_Polished_20261028.zip", preferred_artifact("output/latex/G-M10_A1_Overleaf_Improve_20261028.zip", "output/latex/G-M10_A1_Overleaf_Final.zip"))
SLIDES = preferred_artifact("output/presentation/G-M10_A1_Slides_Polished_20261028.pptx", preferred_artifact("output/presentation/G-M10_A1_Slides_Improve_20261028.pptx", "output/presentation/G-M10_A1_Slides_Final.pptx"))
SLIDES_PDF = preferred_artifact("output/presentation/G-M10_A1_Slides_Polished_20261028.pdf", preferred_artifact("output/presentation/G-M10_A1_Slides_Improve_20261028.pdf", "output/presentation/G-M10_A1_Slides_Final.pdf"))
SCRIPT = preferred_artifact("docs/a1_submission/VIDEO_SCRIPT_IMPROVE_20261028_VI.md", "docs/a1_submission/VIDEO_SCRIPT_FINAL_VI.md")
REPORT = preferred_artifact("output/pdf/G-M10_A1_Report_Confirmed_20261030.pdf", REPORT)
TEX = preferred_artifact("output/latex/G-M10_A1_Overleaf_Confirmed_20261030.zip", TEX)
SLIDES = preferred_artifact("output/presentation/G-M10_A1_Slides_Confirmed_20261030.pptx", SLIDES)
SLIDES_PDF = preferred_artifact("output/presentation/G-M10_A1_Slides_Confirmed_20261030.pdf", SLIDES_PDF)
SCRIPT = preferred_artifact("docs/a1_submission/VIDEO_SCRIPT_CONFIRMED_20261030_VI.md", SCRIPT)
REPORT = preferred_artifact("output/pdf/G-M10_A1_Report_ClassPolicy_20261030.pdf", REPORT)
TEX = preferred_artifact("output/latex/G-M10_A1_Overleaf_ClassPolicy_20261030.zip", TEX)
SLIDES = preferred_artifact("output/presentation/G-M10_A1_Slides_ClassPolicy_20261030.pptx", SLIDES)
SLIDES_PDF = preferred_artifact("output/presentation/G-M10_A1_Slides_ClassPolicy_20261030.pdf", SLIDES_PDF)
SCRIPT = preferred_artifact("docs/a1_submission/VIDEO_SCRIPT_CLASSPOLICY_20261030_VI.md", SCRIPT)
REPORT = preferred_artifact("output/pdf/G-M10_A1_Report_Reviewed_20261104.pdf", REPORT)
TEX = preferred_artifact("output/latex/G-M10_A1_Overleaf_Reviewed_20261104.zip", TEX)
SLIDES = preferred_artifact("output/presentation/G-M10_A1_Slides_Reviewed_20261104.pptx", SLIDES)
SLIDES_PDF = preferred_artifact("output/presentation/G-M10_A1_Slides_Reviewed_20261104.pdf", SLIDES_PDF)
SCRIPT = preferred_artifact("docs/a1_submission/VIDEO_SCRIPT_REVIEWED_20261104_VI.md", SCRIPT)


def esc(value):
    return html.escape(str(value), quote=True)


def source(path, directory=False):
    other = {"assignment2.md": "assignment2/a2_m1/assignment2.md",
             "assignment3.md": "assignment3/a3_m1/assignment3.md"}
    target = other.get(path, "assignment1/a1_m2/" + path)
    return f"{REPO}/{'tree' if directory else 'blob'}/main/{quote(target, safe='/')}"


def table(headers, rows, caption, numeric=(), highlight=None):
    head = "".join(f'<th scope="col" class="{"numeric" if i in numeric else ""}">{esc(x)}</th>' for i, x in enumerate(headers))
    body = []
    for row in rows:
        cells = "".join(f'<{"th scope=" + chr(34) + "row" + chr(34) if i == 0 else "td"} class="{"numeric" if i in numeric else "name" if i == 0 else ""}">{esc(x)}</{"th" if i == 0 else "td"}>' for i, x in enumerate(row))
        body.append(f'<tr class="{"highlight" if row[0] == highlight else ""}">{cells}</tr>')
    return f'<div class="table-wrap" role="region" aria-label="{esc(caption)}" tabindex="0"><table><caption>{esc(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def figure(data, key, caption, compact=False):
    path = data["figure_paths"][key]
    payload = (ROOT / path).read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Expected PNG figure: " + path)
    width, height = struct.unpack(">II", payload[16:24])
    return f'<figure class="figure {"figure-compact" if compact else ""}"><a href="{esc(path)}" aria-label="Open full-size figure: {esc(caption)}"><img src="{esc(path)}" alt="{esc(caption)}" width="{width}" height="{height}" loading="lazy" decoding="async"></a><figcaption>{esc(caption)}</figcaption></figure>'


def resources():
    items = [(REPORT, "Final report", "PDF"), (TEX, "Editable LaTeX project", "ZIP"), (SLIDES, "Presentation slides", "PowerPoint"),
             (SLIDES_PDF, "Slides for reading", "PDF"), (source(SCRIPT), "Presentation script in Vietnamese", "Source"),
             (source("notebooks/A1_Optimization_A100.ipynb"), "Student-run A100 notebook", "Notebook"),
             (source("docs/a1_submission/REPRODUCIBILITY_FINAL.md"), "Reproduction and checkpoint instructions", "Guide"),
             (source("output/a1_final/evidence", directory=True), "Task evidence and 15 final checkpoints", "Source"),
             ("output/a1_final/data/publication_data.json", "Verified results and figure data", "JSON")]
    items.extend(extension_documents())
    return '<ul class="resource-list">' + "".join(f'<li><a href="{esc(path)}">{esc(label)} →</a><span>{esc(kind)}</span></li>' for path, label, kind in items) + '</ul>'


def extension_documents():
    candidates = [
        ("confirmatory-results.html", "Matched training and prospective RGB audit", "Results"),
        ("notebooks/CO3133_Confirmatory_A100.ipynb", "Matched confirmation notebook", "Notebook"),
        ("docs/a1_extension_research/CONFIRMATORY_RESULTS_20261030.md", "Completed follow-up methods and full tables", "Study"),
        ("notebooks/GarmentLab_A100_ImproveToday.ipynb", "GarmentLab controlled-study notebook and recorded results", "Notebook"),
        ("docs/garmentlab/REPRODUCIBILITY.md", "GarmentLab methods, reproduction and local service", "Guide"),
    ]
    return [(path, label, kind) for path, label, kind in candidates if (ROOT / path).is_file()]


def team():
    return '''<div class="team-list"><div><h3><a href="https://github.com/n1velo">Trần Gia Lâm</a></h3><p>2352670</p><p>Experiment execution, model comparison, result analysis and final integration.</p></div><div><h3><a href="https://github.com/cau26">Nguyễn Hữu Cầu</a></h3><p>2352129</p><p>Data analysis, EDA, MLP and assigned technical sections.</p></div></div>'''


def frame(title, description, body, study=False):
    active_home = '' if study else ' aria-current="page"'
    active_study = ' aria-current="page"' if study else ''
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)} · G-M10</title><meta name="description" content="{esc(description)}"><meta name="theme-color" content="#243b45"><link rel="stylesheet" href="assets/site/style.css"></head>
<body><a class="skip-link" href="#main">Skip to content</a><header class="site-header"><div class="container masthead"><a class="brand" href="index.html">G–M10<span>CO3133 · 261</span></a><nav class="site-nav" aria-label="Main navigation"><a href="index.html"{active_home}>Project</a><a href="assignment1-final.html"{active_study}>Assignment 1</a><a href="#materials">Materials</a><a href="{REPO}">Source ↗</a></nav></div></header>
<main id="main">{body}</main><footer class="site-footer"><div class="container footer-grid"><p>Ho Chi Minh City University of Technology · VNU-HCM<br>Faculty of Computer Science and Engineering</p><p>CO3133 · Semester 261 · Instructor: Lê Thành Sách<br><a href="{source('AI_USAGE.md')}">AI usage disclosure</a> · <a href="{REPO}">Repository</a></p></div></footer></body></html>
'''


def result_rows(data):
    return [[NAMES[r["model"]], r["candidate_id"], f'{r["accuracy_mean"]*100:.4f} ± {r["accuracy_sample_sd"]*100:.4f}',
             f'{r["macro_f1_mean"]:.5f} ± {r["macro_f1_sample_sd"]:.5f}', f'{r["parameters"]:,}']
            for r in data["final_exploratory_validation"]]


def original_rows(data):
    return [[NAMES[r["model"]], f'{r["accuracy"]:.2%}', f'{r["macro_f1"]:.6f}', f'{r["parameters"]:,}']
            for r in data["original_official_test"]]


def recipe_rows(data):
    rows = []
    for item in data["selected_recipes"]:
        cfg = item["config"]
        schedule = {"cosine": "Cosine", "constant": "Constant", "warmup_cosine": "Warm-up + cosine"}[cfg["schedule"]]
        drop = cfg.get("head_dropout", 0) if item["model"] in ("mlp", "gru") else cfg.get("dropout", 0)
        rows.append([NAMES[item["model"]], item["candidate_id"], f'{cfg["learning_rate"]:.6g}', schedule,
                     f'{cfg["weight_decay"]:g}', f'{drop:g}'])
    return rows


def garmentlab_excerpt():
    """Keep the extension distinct from the from-scratch course comparison."""
    path = ROOT / "output/garmentlab_final/data/publication_data.json"
    if not path.is_file():
        return "", ""
    extension = json.loads(path.read_text(encoding="utf-8"))
    if extension.get("schema_version") != "garmentlab_verified_publication_v1":
        raise ValueError("Unexpected GarmentLab publication schema")
    fine = extension["recipe_summary"]["finetune"]["assessment"]
    frozen = extension["recipe_summary"]["frozen"]["assessment"]
    accepted = extension["fixed_seed_calibrated_assessment"]["acceptance"]
    paragraph = (f"A separate six-class Fashionpedia extension uses pretrained MobileNetV3-Small. "
        f"Across three paired seeds, fine-tuning records assessment Macro-F1 {fine['macro_f1']['mean']:.4f} "
        f"± {fine['macro_f1']['sample_sd']:.4f}, compared with {frozen['macro_f1']['mean']:.4f} "
        f"± {frozen['macro_f1']['sample_sd']:.4f} for a frozen backbone. Assessment has 327 crops from 221 source groups. "
        f"The frozen review policy gives {accepted['accepted_accuracy']:.2%} accepted accuracy at "
        f"{accepted['coverage']:.2%} coverage, below its empirical 95% calibration target.")
    html_content = (f'<p>{esc(paragraph)}</p><p><a href="garmentlab-results.html">Read the real-image results, '
        'calibration and export checks →</a>. The extension has its own data, model and release evidence.</p>')
    markdown = ("## GarmentLab real-image extension\n\n" + paragraph + "\n\n"
        "[Results page](garmentlab-results.html) · [Reproduction and local inference](docs/garmentlab/REPRODUCIBILITY.md) "
        "· [Verified extension data](output/garmentlab_final/data/publication_data.json)\n\n")
    improved = garmentlab_v2_summary()
    if improved:
        selected = improved["candidate_summary"]["C4"]["assessment_regression"]["macro_f1"]
        anchor = improved["candidate_summary"]["C0"]["assessment_regression"]["macro_f1"]
        accepted_v2 = improved["deployment_calibrated_assessment"]["acceptance"]
        update = (f"The subsequent 20-fit controlled study selected dropout 0.4 (C4) on fresh confirmation seeds. "
            f"On the reused assessment, its mean Macro-F1 is {selected['mean']:.4f} ± {selected['sample_sd']:.4f}, "
            f"versus {anchor['mean']:.4f} ± {anchor['sample_sd']:.4f} for the paired C0 anchor; "
            "this does not establish a classification gain. A disjoint calibration procedure gives the preset C4 seed314 "
            f"{accepted_v2['accepted_accuracy']:.2%} accepted accuracy at {accepted_v2['coverage']:.2%} coverage. "
            "The statistical risk bound is conditional on assumptions, not a future-accuracy guarantee.")
        if (ROOT / "verification/GARMENTLAB_IMPROVE_LOCAL_PYTORCH_SERVICE_20261028.json").is_file():
            update += (" The PyTorch CPU demo passes the expanded numerical and service checks. "
                "ONNX passes the original 64-case audit but fails the broader 327-crop 1e-4 logit criterion; "
                "it remains experimental and is not the primary serving backend.")
        html_content += (f'<p>{esc(update)}</p><p><a href="garmentlab-results.html#controlled-study-v2">'
                         'Controlled study, uncertainty and calibration →</a></p>')
        markdown += (update + f"\n\n[Methods and results]({REPORT}) · "
                     "[Recorded v2 summary](output/garmentlab_improve_return_20261028/FINAL_SUMMARY_V2.json)\n\n")
    return html_content, markdown


def garmentlab_v2_summary():
    path = ROOT / "output/garmentlab_improve_return_20261028/FINAL_SUMMARY_V2.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("schema") != "garmentlab_improved_study_summary_v2"
            or result.get("smoke_only") is not False
            or result.get("status") != "COMPLETE_RESEARCH_ARTIFACTS"
            or result.get("deployment_task_id") != "final_C4_seed314"):
        raise ValueError("Expected the recorded complete GarmentLab v2 C4 study")
    return result


def garmentlab_v2_section():
    """Render the completed second study without replacing historical v1 evidence."""
    result = garmentlab_v2_summary()
    if result is None:
        return ""
    rows = []
    for candidate, label in [("C0", "C0 · dropout 0.2 anchor"), ("C4", "C4 · dropout 0.4 selected")]:
        entry = result["candidate_summary"][candidate]
        rows.append([label, f"{entry['selection']['macro_f1']['mean']:.5f} ± {entry['selection']['macro_f1']['sample_sd']:.5f}",
                     f"{entry['assessment_regression']['accuracy']['mean']*100:.2f} ± {entry['assessment_regression']['accuracy']['sample_sd']*100:.2f}%",
                     f"{entry['assessment_regression']['macro_f1']['mean']:.5f} ± {entry['assessment_regression']['macro_f1']['sample_sd']:.5f}"])
    comparison = table(["Recipe", "Selection Macro-F1", "Reused assessment accuracy", "Reused assessment Macro-F1"], rows,
        "Controlled v2 final fits · mean ± sample SD across seeds 314, 2718 and 5772", numeric=(1, 2, 3))
    raw = result["assessment"][result["deployment_task_id"]]
    calibrated = result["deployment_calibrated_assessment"]
    accepted = calibrated["acceptance"]
    calibration = result["calibration_v2"]
    selected_bound = calibration["threshold_selection"]["selected"]
    interval = result["grouped_uncertainty"]["paired_mean_macro_f1_delta_interval"]
    parity = result["gates"]["parity"]
    local_path = ROOT / "verification/GARMENTLAB_IMPROVE_LOCAL_PYTORCH_SERVICE_20261028.json"
    local = json.loads(local_path.read_text(encoding="utf-8")) if local_path.is_file() else None
    serving_note = ""
    if local:
        pytorch_check = local["pytorch_full327_numerical_regression"]
        onnx_check = local["onnx_expanded_regression"]
        soak = next(item["summary"] for item in local["real_http_phases"] if item["kind"] == "soak_supported_concurrency")
        serving_note = (f'<p><strong>The broader Windows check passes PyTorch and fails ONNX.</strong> '
            f'Across all 327 assessment crops at batch 1, local PyTorch differs from the archived Colab outputs by at most '
            f'{pytorch_check["maximum_absolute_error"]:.8f}, within the unchanged 0.0001 limit. The preserved ONNX artifact '
            f'reaches {onnx_check["maximum_absolute_error"]:.8f}, exceeding that limit. Both produce zero measured label or review-decision '
            'differences. Three additional CPU exporter/runtime variants also fail the broader criterion. '
            'The original archive is unchanged; its recorded 64-case pass is retained with that scope. '
            '<strong>ONNX remains experimental and is not the primary serving backend.</strong></p>'
            '<p>The functional demo uses <strong>PyTorch CPU</strong>. Its seven PNG checks and in-process HTTP contracts pass. '
            f'A separate 60-second loopback soak completes {soak["requests"]:,}/{soak["requests"]:,} HTTP 200 predictions '
            f'at concurrency 2, with p95 = {soak["successful_prediction_client_e2e"]["p95_ms"]:.2f} ms. '
            'These are repeated calls on two JPEG/crop cases; they do not measure new-image quality, a hosted SLA or production reliability. '
            'They must not be compared with historical v1 timings as a controlled speedup.</p>'
            f'<p><a href="{source("verification/GARMENTLAB_IMPROVE_LOCAL_PYTORCH_SERVICE_20261028.json")}">'
            'PyTorch service and broader ONNX regression</a> · '
            f'<a href="{source("verification/GARMENTLAB_IMPROVE_LOCAL_DERIVED_CPU_20261028.json")}">'
            'Preserved unsuccessful ONNX variants</a></p>')
    calibration_table = table(["Metric", "C4 seed314 raw", "Temperature scaled"],
        [[label, f"{raw[key]:.4f}", f"{calibrated[key]:.4f}"]
         for key, label in [("nll", "NLL"), ("ece", "ECE, 15 bins"), ("brier_score", "Brier score")]],
        "Preset C4 seed 314 · descriptive results on the reused 327-crop assessment", numeric=(1, 2))
    return f'''<section id="controlled-study-v2" class="landing-section container">
<p class="eyebrow">Completed controlled study · 28 October 2026</p><h2>More control; no demonstrated F1 gain</h2>
<p>The second A100 study completed 20 fits × 18 epochs: eight discovery fits, six fresh-seed confirmation fits and six paired final fits. All candidates start from ImageNet weights. C4 changes the classifier dropout from 0.2 to 0.4; the other training settings match C0. The recipe is selected by mean selection Macro-F1 on confirmation seeds 202 and 303. Final seeds 314, 2718 and 5772 were fixed in advance, as was deployment seed 314.</p>
{comparison}
<p>The selected recipe wins the confirmation criterion narrowly, but its assessment mean Macro-F1 is lower than the anchor. The paired component-bootstrap interval for C4 minus C0 is [{interval[0]:+.4f}, {interval[1]:+.4f}], which includes zero. This is a descriptive interval over 221 source/duplicate components with fixed models and seeds, based on 2,000 resamples. It is not a fresh-test or post-selection-adjusted confidence claim.</p>
<p>Assessment remains 327 crops and had already been inspected in v1. The v2 scores are explicitly <strong>post-test exploratory regression</strong>. Model and threshold choices are frozen before this evaluation. The selected release stays C4; the assessment result is reported without changing that prior decision.</p>
<h3>Disjoint calibration and a stricter review policy</h3>
<p>Calibration is split by the union of source and duplicate components: 621 components/936 crops fit temperature; 622 components/899 crops select the review threshold. One metadata-selected representative per threshold component defines the risk population. A fixed 101-threshold grid uses one-sided exact binomial bounds with Bonferroni correction. The active threshold 0.88 accepts {selected_bound['accepted_count']}/622 representatives with {selected_bound['error_count']} errors; the simultaneous error upper bound is {selected_bound['error_upper_bound']:.4%} under the stated assumptions.</p>
{calibration_table}
<p>Temperature {result['calibration']['temperature']:.4f} preserves class predictions. On reused assessment, the frozen policy accepts {accepted['accepted_count']}/327 crops, with {accepted['accepted_correct']} correct: <strong>{accepted['accepted_accuracy']:.2%} accepted accuracy at {accepted['coverage']:.2%} coverage</strong>. Historical v1 accepted accuracy was 92.54% at 69.72% coverage. The policies use different models and calibration procedures, so this is an observed trade-off rather than an isolated causal comparison.</p>
<p>The calibration set was previously inspected in v1. Independence and future distribution assumptions have not been established, and equally weighted component representatives differ from crop-weighted traffic. No unconditional 95% accuracy guarantee is claimed.</p>
<h3>PyTorch serving and an ONNX limitation</h3>
<p>The returned C4 manifest records ready status after its original data, selection-quality, calibration, 64-case ONNX audit and serving protocol. The original ONNX maximum absolute logit error is {parity['maximum_absolute_error']:.8f}, below 0.0001, with zero top-1 disagreements at the recorded batch sizes 1, 3 and 64. These statements describe that original protocol; they do not override subsequent broader verification.</p>
{serving_note}
<p>Separately, an isolated Windows CPU environment passed dependency and service checks. A three-minute local HTTP soak completed 10,506/10,506 successful requests with p95 = 37.26 ms. That soak used the historical <strong>v1 CPU model</strong> and two repeated input cases; it is not a v2 benchmark or hosted SLA. Request logs retain request identifiers, status and duration without storing uploaded image contents.</p>
<p><a href="{REPORT}">Report: methods and results</a> · <a href="output/garmentlab_improve_return_20261028/FINAL_SUMMARY_V2.json">Recorded v2 summary</a> · <a href="output/garmentlab_improve_return_20261028/release_candidate/release.json">Exact C4 release manifest</a> · <a href="{source('docs/a1_extension_research/CALIBRATION_V2_METHOD.md')}">Calibration assumptions</a> · <a href="{source('docs/garmentlab/REPRODUCIBILITY.md')}">Run the local application</a></p>
</section>'''


def add_garmentlab_v2_section(content):
    """Idempotently append the current study before preserved historical sections."""
    import re
    content = re.sub(r'<section id="controlled-study-v2".*?</section>\n?', '', content, flags=re.S)
    section = garmentlab_v2_section()
    if section:
        marker = '<section class="landing-section container"><h2>Four data roles.'
        if marker not in content:
            raise ValueError("Expected historical GarmentLab page insertion point")
        content = content.replace(marker, section + '\n' + marker, 1)
        content = content.replace('<h2>Four data roles.</h2>', '<h2>Historical v1 study: four data roles</h2>', 1)
        content = content.replace(
            '<p class="lead">A paired three-seed comparison of a frozen ImageNet backbone and fine-tuning on six Fashionpedia garment classes, followed by calibration, stress tests and an audited export.</p>',
            '<p class="lead">Six-class garment crop classification with MobileNetV3-Small: a paired baseline comparison, a completed controlled regularization study, disjoint calibration and verified local serving.</p>', 1)
        content = content.replace('>Method and results</a>', '>Historical v1 method and results</a>', 1)
    return content


def build_pages(data):
    final_table = table(["Model", "Recipe", "Accuracy (%)", "Macro-F1", "Parameters"], result_rows(data),
                        "Final exploratory validation · mean ± sample SD over seeds 42, 43 and 44", numeric=(2, 3, 4), highlight="CNN")
    old_table = table(["Model", "Test accuracy", "Test macro-F1", "Parameters"], original_rows(data),
                      "Original frozen comparison · 10,000 official test images · one seed", numeric=(1, 2, 3), highlight="CNN")
    cnn = next(row for row in data["final_exploratory_validation"] if row["model"] == "cnn")
    hours = data["timing"]["active_controller_clock"]["active_run_seconds"] / 3600
    video = '<p class="video-status">Presentation recording: not published.</p>'
    landing = f'''<section class="hero container"><div class="hero-grid"><div><p class="eyebrow">Deep Learning and Its Applications</p><h1>From pixels to<br>sequence models.</h1><p class="lead">A comparative study of five PyTorch architectures for Fashion-MNIST. We examine what representation, training recipes and computational cost change—and what the evidence can establish.</p><div class="actions"><a class="button" href="assignment1-final.html">Read the study</a><a class="text-link" href="{REPORT}">Download the report →</a></div></div><dl class="hero-meta"><div><dt>Institution</dt><dd>HCMUT · VNU-HCM<br>Computer Science and Engineering</dd></div><div><dt>Course</dt><dd>CO3133 · Semester 261</dd></div><div><dt>Instructor</dt><dd>Lê Thành Sách</dd></div><div><dt>Authors</dt><dd>Trần Gia Lâm<br>Nguyễn Hữu Cầu</dd></div></dl></div></section>
<div class="container"><dl class="facts"><div><dt>5</dt><dd>architecture families</dd></div><div><dt>61</dt><dd>completed optimization tasks</dd></div><div><dt>3</dt><dd>final training seeds</dd></div><div><dt>A100</dt><dd>recorded GPU</dd></div></dl></div>
<section class="landing-section container"><div class="section-intro"><div><p class="eyebrow">Assignment 1 · final study</p><h2>One task.<br>Five representations.</h2></div><p>Flattened pixels, nonlinear features, local convolutions, ordered rows and patch attention share a common image-classification pipeline. A frozen candidate search compares complete training recipes, followed by three-seed fitting of each selected recipe.</p></div>{figure(data,'final_f1_seeds','Exploratory validation macro-F1 for the selected recipes across three training seeds. The same previously seen 6,000-image validation set selects each checkpoint.')}<p>The selected CNN records macro-F1 <strong>{cnn['macro_f1_mean']:.5f} ± {cnn['macro_f1_sample_sd']:.5f}</strong> across three final seeds. These are validation scores, not new blind test results.</p><a class="text-link" href="assignment1-final.html#results">Explore the results and their limits →</a></section>
<section class="landing-section container"><div class="section-intro"><div><p class="eyebrow">Separate evidence</p><h2>The original<br>official-test comparison.</h2></div><p>Before the optimization study, five 20-epoch models were selected by validation loss and frozen together. Their official-test scores remain a separate, unchanged comparison.</p></div>{old_table}<p class="caption">CNN has the highest observed test score in this single-seed comparison. The later optimization study does not supply a replacement test score.</p></section>
<section id="materials" class="landing-section container"><div class="section-intro"><div><p class="eyebrow">Read and reproduce</p><h2>Study materials.</h2></div><p>The report develops the method, comparison and error analysis. Source code, configurations and checkpoint reconstruction instructions make the results traceable to the recorded runs.</p></div>{resources()}{video}</section>
<section class="landing-section container"><p class="eyebrow">Course projects</p><h2>Assignment index.</h2><ul class="resource-list"><li><a href="assignment1-final.html">Assignment 1 · Fashion-MNIST study →</a><span>Final study</span></li><li><a href="{source('assignment1.md')}">Assignment 1 · M1 archive ↗</a><span>Repository</span></li><li><a href="{source('assignment2.md')}">Assignment 2 · Large-scale data and specialized tasks ↗</a><span>Repository</span></li><li><a href="{source('assignment3.md')}">Assignment 3 · Multimodal deep learning ↗</a><span>Repository</span></li></ul></section>
<section class="landing-section container"><p class="eyebrow">Group G-M10</p><h2>Authors and contributions.</h2>{team()}<h3>AI usage</h3><p class="small muted">AI assisted LaTeX writing and pipeline development. The <a href="{source('AI_USAGE.md')}">disclosure</a>.</p></section>'''
    cost_rows = []
    for row in data["final_exploratory_validation"]:
        selected = [r for r in data["final_per_seed"] if r["model"] == row["model"]]
        latency = statistics.mean(r["inference"]["per_image_ms_amortized"] for r in selected)
        cost_rows.append([NAMES[row["model"]], f'{row["parameters"]:,}', f'{row["fit_wall_seconds_mean"]:.2f}', f'{latency:.6f}'])
    paired_rows = [[NAMES[r["model"]], f'C0 → {r["selected_id"]}', f'{r["delta_f1_pp"][0]:+.4f}',
                    f'{r["delta_f1_pp"][1]:+.4f}', f'{r["mean_delta_f1_pp"]:+.4f}'] for r in data["inner_paired_selection"]]
    error = next(r for r in data["per_class"] if r["model"] == "cnn" and r["seed"] == 42)
    error_rows = [[r["true_class"], r["predicted_class"], str(r["count"]), f'{r["rate_within_true_class"]:.2%}'] for r in error["top_confusions"][:5]]
    architecture_rows = [["Linear", "784 flattened pixels → 10 logits", "7,850"], ["MLP", "784 → 256 → ReLU → 10 logits", "203,530"],
                         ["CNN", "Three convolution blocks, pooling and global spatial mean", "93,962"],
                         ["GRU", "28 ordered rows, 28 inputs per step, hidden size 128", "61,962"],
                         ["Transformer", "49 projected 4×4 patches, positions and two attention encoder layers", "274,954"]]
    sections = [
        ("problem", "Problem and data", f'''<p>Given a 28×28 grayscale product image, predict one of ten Fashion-MNIST categories. Closely related upper-body garments make silhouette and boundary information important; accuracy alone hides these class-level errors.</p><p>The original 60,000 training images are partitioned into 54,000 training and 6,000 validation images with stratification and seed 42. The official 10,000-image test partition is separate. All inputs use float32 values in [0,1], with no augmentation or pretrained weights.</p>{figure(data,'dataset_overview','Fashion-MNIST categories and class-balanced partitions. The inner 48,600/5,400 split is drawn only from the original 54,000 training images.')}<p>For recipe selection, an inner split uses 48,600 images for training and 5,400 for validation. Final selected recipes return to 54,000 training images and the previously seen 6,000-image validation set. The latter selects the reported final checkpoints.</p><p class="small muted">Source: <a href="https://github.com/zalandoresearch/fashion-mnist">Zalando Research Fashion-MNIST</a>, MIT license. Split indices and dataset content fingerprints are retained. Disjoint image IDs do not establish that product-level or near-duplicate leakage is absent.</p>'''),
        ("methods", "Architectures and training", f'''<p>Each model produces ten logits for cross-entropy loss. PyTorch supplies the underlying neural-network layers and automatic differentiation; the project implements composition, data handling, training, evaluation and experiment control.</p>{table(['Model','Input representation and structure','Parameters'],architecture_rows,'Five models trained without pretrained weights',numeric=(2,))}<p>All optimization tasks use batch size 128, 60 epochs, gradient clipping at norm 1.0 and the same fixed tensor shapes. Optimizer, learning rate, dropout and scheduling form the recipe being compared. This is a complete-recipe study, not an architecture-only causal ablation.</p>{table(['Model','Recipe','Learning rate','Schedule','Weight decay','Dropout'],recipe_rows(data),'Selected training recipes',numeric=(2,4,5))}<p class="caption">Dropout denotes the hidden-state dropout for MLP/GRU and configured dropout for CNN/Transformer. Full configurations are available in the <a href="{source('configs/a1_optimization_v3.json')}">protocol</a> and <a href="{source('a1opt',True)}">optimization source</a>.</p>'''),
        ("protocol", "A fixed selection protocol", f'''<ol class="protocol-steps"><li><strong>Discover</strong><p>36 frozen candidates. Train each for 60 epochs on the inner split, using seed 101. Choose one non-anchor challenger per family.</p></li><li><strong>Confirm</strong><p>10 tasks compare each challenger with C0 on fresh seed 202. The locked selection rule averages selected inner macro-F1 across seeds 101 and 202.</p></li><li><strong>Fit and report</strong><p>15 tasks fit five selected recipes with seeds 42, 43 and 44 on 54,000 images. Report checkpoint-selected scores on the old 6,000-image validation set.</p></li></ol><p>Optuna 4.8.0 RandomSampler supplies a fixed candidate set; it is not an adaptive or global-optimum search. Linear has four candidates and each other family eight. Each epoch is completed; there is no pruning or early stopping. Checkpoints maximize validation macro-F1, then minimize cross-entropy, then retain the earliest epoch.</p>{figure(data,'discovery_confirmation','Candidate discovery and the anchor/challenger confirmation stage on inner validation.')}<h3>What confirmation supports</h3>{table(['Model','Anchor → selected','Seed 101 ΔF1 (pp)','Seed 202 ΔF1 (pp)','Mean ΔF1 (pp)'],paired_rows,'Paired inner-validation differences against C0',numeric=(2,3,4))}{figure(data,'paired_confirmation_deltas','Paired inner-validation macro-F1 differences. Seed 101 also selected each challenger; seed 202 is the single fresh confirmation seed.')}<p>The Transformer challenger gains on seed 101 but loses on fresh seed 202. It is retained by the declared two-seed mean rule. That rule is a selection heuristic, not a significance test; reuse of discovery seed 101 leaves selection bias.</p>'''),
        ("results", "Final exploratory validation", f'''{final_table}<div class="evidence-note"><p><strong>Read these as selected-validation scores.</strong> The same old 6,000 images select the checkpoints and provide these measurements. Sample SD uses three training seeds on one fixed split; it is neither a confidence interval nor uncertainty over unseen datasets.</p></div>{figure(data,'final_f1_seeds','Individual final training seeds and their mean macro-F1. CNN C5 records the highest observed mean in this validation comparison.')}<p>CNN combines local spatial features with 93,962 parameters and has the highest observed mean macro-F1. GRU has the smallest nonlinear parameter count. The closely grouped MLP, GRU and Transformer results should be interpreted with their training-seed variation and recipe differences.</p>{figure(data,'learning_curves','Final selected-recipe learning curves over all 60 epochs. The checkpoint rule can retain an earlier epoch.')}<p>Final selected epochs vary across seeds: CNN selects epochs 44, 60 and 55; GRU selects 37, 28 and 34. A fixed epoch budget does not imply convergence, and the final epoch is not always the selected checkpoint.</p>'''),
        ("cost", "Accuracy and computation", f'''{table(['Model','Parameters','Mean fit (s)','Forward (ms/image)'],cost_rows,'Final three-seed recorded computation',numeric=(1,2,3))}{figure(data,'accuracy_cost','Validation accuracy in relation to parameter count and recorded forward cost for the final selected recipes.')}<p>Fit time includes completed training and validation loops plus in-memory checkpoint selection. It excludes transaction writes and finalization. Forward latency is batch-amortized: 128 already-on-GPU images, ten warm-up passes and fifty measured passes, excluding loading and transfers.</p><p>The complete controller records {hours:.2f} active hours within a seven-day upper cap. The cap is a limit, not a claim that seven days were consumed. The recorded GPU is NVIDIA A100-SXM4-40GB with PyTorch 2.11.0+cu130 and CUDA 13.0.</p>'''),
        ("errors", "Where the model still fails", f'''<p>For CNN C5, seed 42, the selected validation checkpoint predicts {error['total_correct']:,} of 6,000 images correctly. Shirt remains a difficult category: related outlines lead to confusion with T-shirt/top, Coat and Pullover.</p>{figure(data,'cnn_seed42_confusion','CNN C5, seed 42: confusion counts on the 6,000-image old validation set. Rows are true classes; columns are predicted classes.',True)}{table(['True class','Predicted class','Images','Within true class'],error_rows,'Largest directed validation errors · CNN C5, seed 42',numeric=(2,3))}{figure(data,'cnn_seed42_examples','CNN C5, seed 42: eight correct examples in original-ID order and eight errors with the largest supplied true-label NLL. This is an illustrative selection, not a random sample.')}<p>Shape similarity is a plausible explanation, not a verified cause. Class-specific evaluation, alternative regularization and data-focused inspection would be useful next investigations. They have not been run as additional results here.</p>'''),
        ("official-test", "Original official-test results", f'''<p>The earlier five-model experiment used seed 42 and a 20-epoch budget. Each checkpoint was chosen by minimum validation cross-entropy, and all five were frozen before evaluation on the official test partition.</p>{old_table}<p>These measured test results remain unchanged. The later 60-epoch optimization uses different recipes, a different checkpoint criterion and selected validation data, so its scores cannot be read as test-set gains over this table.</p>'''),
        ("materials", "Materials and reproduction", f'''{resources()}{video}<h3>Run a new optimization study</h3><pre><code>python -m pip install -r requirements-a1-optimization.txt
python tools/run_a1_optimization.py --stage plan --out outputs/opt3_reproduction --device cuda
python tools/run_a1_optimization.py --stage smoke --out outputs/opt3_reproduction --device cuda
python tools/run_a1_optimization.py --stage run --out outputs/opt3_reproduction --device cuda
python tools/run_a1_optimization.py --stage export --out outputs/opt3_reproduction</code></pre><p>Use an A100 runtime with compatible PyTorch/torchvision. Resume an interrupted task by rerunning into the same output directory with its recorded source and environment. Start an independent experiment in a new directory. The optimization runner does not open or evaluate the official test partition.</p><p class="small muted">Evidence: 715 export-manifest hashes, 61 completed tasks and 122 strict CPU checkpoint loads were verified. All 338,400 saved prediction rows were checked in the independent statistics review. <a href="{source('verification/OPT3_RETURN_REVIEW_1d2d83edf1dc.md')}">Artifact audit</a> · <a href="{source('verification/OPT3_INDEPENDENT_STATISTICS_1d2d83edf1dc.json')}">Statistics audit</a> · <a href="{source('README.md')}">Repository setup guide</a>.</p>'''),
        ("limits", "Conclusions and limits", f'''<p>This project shows how representation and training choices interact in a controlled small-image task. CNN is the highest-scoring observed recipe in both evidence layers; GRU is compact, and model size alone does not determine runtime. The study supports those bounded comparisons, not a universal ranking of model families.</p><ul><li>The final validation set was previously seen and selects the checkpoint; no new blind test was performed.</li><li>Only one fresh seed confirms each challenger, and discovery-seed reuse can favor a selected candidate.</li><li>Three final seeds describe stochastic training variation on a fixed split.</li><li>Fresh matched final C0 anchors are absent; historical comparisons mix recipe, engine and checkpoint-rule changes.</li><li>Raw predictions support independent scores, but logs and checksums do not replace scrutiny of assumptions or human understanding.</li></ul><h3>Authors and AI disclosure</h3>{team()}<p class="small muted">AI assisted LaTeX writing and pipeline development. <a href="{source('AI_USAGE.md')}">AI usage disclosure</a>.</p><h3>Sources</h3><ol class="reference-list"><li><a href="https://arxiv.org/abs/1708.07747">Xiao, Rasul and Vollgraf. Fashion-MNIST (2017).</a></li><li><a href="https://github.com/zalandoresearch/fashion-mnist">Zalando Research. Dataset, labels and license.</a></li><li><a href="https://arxiv.org/abs/1406.1078">Cho et al. RNN Encoder–Decoder (2014).</a></li><li><a href="https://arxiv.org/abs/1706.03762">Vaswani et al. Attention Is All You Need (2017).</a></li></ol>'''),
    ]
    extension_html, _ = garmentlab_excerpt()
    if extension_html:
        landing = landing.replace('<section id="materials"',
            '<section class="landing-section container"><p class="eyebrow">Real-image extension</p>'
            '<h2>GarmentLab.</h2>' + extension_html + '</section><section id="materials"', 1)
        sections.insert(-1, ("garmentlab", "GarmentLab real-image extension", extension_html))
    toc = '<aside class="toc"><nav aria-label="On this page"><p>On this page</p><ol>' + ''.join(f'<li><a href="#{key}">{esc(title)}</a></li>' for key,title,_ in sections) + '</ol></nav></aside>'
    article = ''.join(f'<section id="{key}" class="study-section"><h2>{esc(title)}</h2>{content}</section>' for key,title,content in sections)
    study = f'''<section class="hero container"><p class="eyebrow">CO3133 · Assignment 1 · Group G-M10</p><h1>Five architectures.<br>One classification task.</h1><p class="lead">From linear models to sequence models: a reproducible Fashion-MNIST comparison, followed by a fixed recipe search and three-seed validation study.</p><div class="actions"><a class="button" href="{REPORT}">Read the final report</a><a class="text-link" href="{SLIDES_PDF}">View the slides →</a></div></section><div class="container article-layout">{toc}<article>{article}</article></div>'''
    return frame("Deep Learning and Its Applications", "Five PyTorch architecture families, a frozen test comparison and a verified 61-task Fashion-MNIST study by Group G-M10.", landing), frame("Fashion-MNIST · Final study", "Methods, verified results, error analysis and reproducibility for the G-M10 five-model Fashion-MNIST project.", study, True)


def markdown_pages(data):
    final = md_table(["Model", "Recipe", "Accuracy (%) mean ± SD", "Macro-F1 mean ± SD", "Parameters"], result_rows(data))
    original = md_table(["Model", "Official-test accuracy", "Official-test macro-F1", "Parameters"], original_rows(data))
    materials = f"- [Final report PDF]({REPORT})\n- [Editable LaTeX project]({TEX})\n- [Slides PowerPoint]({SLIDES}) · [Slides PDF]({SLIDES_PDF})\n- [Vietnamese presentation script]({SCRIPT})\n- [A100 notebook](notebooks/A1_Optimization_A100.ipynb)\n- [Reproduction and checkpoint loading](docs/a1_submission/REPRODUCIBILITY_FINAL.md)\n- [Verified publication data](output/a1_final/data/publication_data.json)"
    if extension_documents():
        materials += "\n" + "\n".join(f"- [{label}]({path})" for path, label, _ in extension_documents())
    commands = '''```bash
python -m pip install -r requirements-a1-optimization.txt
python tools/run_a1_optimization.py --stage plan --out outputs/opt3_reproduction --device cuda
python tools/run_a1_optimization.py --stage smoke --out outputs/opt3_reproduction --device cuda
python tools/run_a1_optimization.py --stage run --out outputs/opt3_reproduction --device cuda
python tools/run_a1_optimization.py --stage export --out outputs/opt3_reproduction
```'''
    readme = f'''# CO3133 · Deep Learning and Its Applications

Group **G-M10**, Semester 261 · Ho Chi Minh City University of Technology, VNU-HCM · Faculty of Computer Science and Engineering · Instructor **Lê Thành Sách**.

We compare five PyTorch architecture families on Fashion-MNIST: Linear, MLP, CNN, GRU and Transformer. The project combines a shared data pipeline, recorded checkpoints, experiment control and analysis of accuracy, class errors and computational cost. All five Fashion-MNIST models train without pretrained weights.

**[Website source](index.html) · [Final study](assignment1-final.md) · [Source repository]({REPO})**

## Team

| Member | Student ID | Main responsibilities |
|---|---|---|
| [Trần Gia Lâm](https://github.com/n1velo) | 2352670 | Team lead; experimental coordination, Colab training, result analysis and final integration. |
| [Nguyễn Hữu Cầu](https://github.com/cau26) | 2352129 | Data analysis, EDA, MLP and assigned technical sections. |

## Final exploratory validation

The optimization study completed **61 tasks × 60 epochs**: 36 discovery, 10 confirmation and 15 final tasks. Each selected recipe is fit with seeds **42, 43 and 44**. Checkpoints maximize validation macro-F1, with lower cross-entropy and then the earliest epoch breaking ties.

{final}

Mean ± sample SD is across three training seeds on the same previously seen 6,000-image validation set. These are checkpoint-selected validation scores, not new official-test results or confidence intervals. CNN C5 has the highest observed mean; the study does not establish a global optimum or universal architecture ranking.

## Original frozen official-test comparison

Before optimization, five 20-epoch models were selected by minimum validation loss and frozen together. They use one training seed and the official 10,000-image test partition.

{original}

This table remains unchanged. Later validation scores use different recipes and checkpoint selection; they cannot be interpreted as official-test gains.

## Read the project

{materials}

Presentation recording: not published.

## Reproduce the optimization study

Use an NVIDIA A100 CUDA runtime with a compatible PyTorch/torchvision pair. The measured environment records PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA 13.0, NumPy 2.1.3 and Optuna 4.8.0. Exact source and environment records accompany the run export.

{commands}

The inner split is 48,600/5,400 images drawn only from the original 54,000 training images. Recipe selection compares an anchor and challenger across discovery seed 101 and fresh confirmation seed 202. The final 54,000/6,000 fits use selected recipes and seeds 42–44. No official-test dataset is opened by this runner.

Rerun the same command and output directory to resume committed work. Use a new directory for an independent experiment and restore the recorded source/environment for exact recovery. Completed task hashes and source/split bindings prevent silent replacement. See the [reproducibility guide](docs/a1_submission/REPRODUCIBILITY_FINAL.md) for reconstruction details.

For the original five-model pipeline, see the [reproducibility guide](docs/a1_submission/REPRODUCIBILITY_FINAL.md) and [configs/a1_final.json](configs/a1_final.json). Exact historical reconstruction uses the source snapshot recorded by its export. Evaluation commands report validation; official-test evaluation is a separate frozen protocol.

## Evidence and limits

- [Artifact/checkpoint review](verification/OPT3_RETURN_REVIEW_1d2d83edf1dc.md): 715 manifest hashes and 122 strict CPU checkpoint loads.
- [Independent statistics review](verification/OPT3_INDEPENDENT_STATISTICS_1d2d83edf1dc.json): 338,400 saved prediction rows, histories, selection and aggregates checked.
- Recorded controller time is 8.34 active hours under a seven-day upper cap; this is not seven days of GPU training.
- Only one fresh seed confirms each challenger; discovery-seed reuse leaves selection bias. Transformer loses to C0 on fresh seed 202 despite winning the declared two-seed mean rule.
- Final training-seed SD does not measure uncertainty over splits or unseen datasets. No matched final C0 anchors isolate causal gains over historical CE-selected models.

## Course assignment index

| Assignment | Focus | Materials |
|---|---|---|
| 1 | Image classification and architecture comparison | [Final study](assignment1-final.md) |
| 2 | Large-scale data and specialized tasks | [Existing assignment page](assignment2.md) |
| 3 | Multimodal deep learning | [Existing assignment page](assignment3.md) |

## AI usage

AI assisted LaTeX writing and pipeline development. [AI_USAGE.md](AI_USAGE.md).

## Local website preview

```bash
python tools/build_a1_pages.py
python -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/`. The static pages use relative URLs and a shared stylesheet, with no installation or frontend build required. Existing M1/A2/A3 source pages remain intact.
'''
    paired = md_table(["Model", "Selected", "Seed 101 ΔF1 (pp)", "Fresh seed 202 ΔF1 (pp)", "Mean ΔF1 (pp)"],
                      [[NAMES[r['model']],r['selected_id'],f"{r['delta_f1_pp'][0]:+.4f}",f"{r['delta_f1_pp'][1]:+.4f}",f"{r['mean_delta_f1_pp']:+.4f}"] for r in data['inner_paired_selection']])
    assignment = f'''# Assignment 1 · Final study

**From Linear Models to Modern Sequence Models: A Comparative Study for Image Classification**

HCMUT · VNU-HCM · Faculty of Computer Science and Engineering

CO3133 · Semester 261 · Instructor Lê Thành Sách · Group G-M10

Trần Gia Lâm (2352670) · Nguyễn Hữu Cầu (2352129)

**[Read the illustrated study](assignment1-final.html) · [Final report]({REPORT}) · [Source]({REPO})**

## Problem and data

Predict one of ten apparel categories from a 28×28 grayscale Fashion-MNIST image. Inputs are float32 [0,1]; no augmentation or pretrained weights are used. The dataset is class-balanced and distributed by [Zalando Research](https://github.com/zalandoresearch/fashion-mnist) under MIT.

The original training partition is split into 54,000 training and 6,000 validation images with stratification and seed 42. The official 10,000-image test partition stays separate. Recipe discovery uses an inner 48,600/5,400 split drawn exclusively from the original 54,000 training images. Final selected recipes are trained on 54,000 images and checkpoint-selected on the old 6,000 validation images.

![Data overview]({data['figure_paths']['dataset_overview']})

## Method and experimental setup

Linear/MLP flatten pixels; CNN extracts spatial features; GRU scans 28 rows; Transformer processes 49 projected patches with positional embeddings and attention. PyTorch provides layers/autograd; this repository implements model composition, data handling, training/evaluation and experiment control.

The fixed Optuna RandomSampler candidate set contains four Linear recipes and eight for each other family. Discovery has 36 tasks with seed 101; confirmation has ten anchor/challenger tasks with fresh seed 202; final fitting has 15 tasks across seeds 42–44. All 61 tasks completed 60 epochs, batch size 128, on A100. The checkpoint criterion is maximum validation macro-F1, then minimum cross-entropy, then earliest epoch. This compares complete recipes; it is not an architecture-only ablation.

{md_table(['Model','Recipe','Learning rate','Schedule','Weight decay','Dropout'],recipe_rows(data))}

![Discovery and confirmation]({data['figure_paths']['discovery_confirmation']})

## Final exploratory validation

{final}

These are mean ± sample SD across three training seeds, evaluated on the same checkpoint-selection validation set. They are not new blind test results or confidence intervals.

![Final seed comparison]({data['figure_paths']['final_f1_seeds']})

CNN C5 gives the highest observed mean macro-F1 with 93,962 parameters. GRU is the smallest nonlinear model. The final checkpoint may precede epoch 60; selected CNN epochs are 44, 60 and 55, and GRU epochs are 37, 28 and 34.

![Learning curves]({data['figure_paths']['learning_curves']})

## Paired recipe selection

{paired}

Seed 101 was used to choose the challenger; seed 202 is the only fresh confirmation seed. The Transformer challenger loses on seed 202 but wins the declared two-seed mean rule. This is a locked heuristic, not a significance test or proof of general superiority.

![Paired confirmation differences]({data['figure_paths']['paired_confirmation_deltas']})

## Cost and errors

![Accuracy and computation]({data['figure_paths']['accuracy_cost']})

Timing uses recorded A100 runs. Forward latency is amortized over batch 128, with ten warm-up and fifty measured forward passes; loading and transfers are excluded. Fit time includes completed train/validation loops and in-memory selection but excludes transaction writes/finalization. The controller recorded 8.34 active hours under a seven-day upper cap.

![CNN seed 42 validation confusion]({data['figure_paths']['cnn_seed42_confusion']})

CNN C5, seed 42, gets 5,597/6,000 old validation images correct. Its largest directed errors include Shirt → T-shirt/top (49/600), T-shirt/top → Shirt (45/600) and Coat → Shirt (42/600). Related silhouettes are a plausible cause to investigate, not a verified explanation.

![CNN selected examples]({data['figure_paths']['cnn_seed42_examples']})

The example selection uses eight correct predictions in original-ID order and eight errors with the largest supplied true-label NLL. It is illustrative rather than random.

## Original official-test evidence

{original}

These unchanged 20-epoch, seed-42 results use minimum-validation-CE checkpoint selection before freeze and official test. The later optimization has different recipes, engine and checkpoint criterion; its selected validation scores must not be described as test gains.

## Reproduction and materials

{materials}

{commands}

Use a compatible A100 CUDA runtime. Resume the same run with the same recorded source/environment; use a new output directory for a new experiment. The optimization runner does not open official test data. [Artifact review](verification/OPT3_RETURN_REVIEW_1d2d83edf1dc.md) and [statistics review](verification/OPT3_INDEPENDENT_STATISTICS_1d2d83edf1dc.json) make the records traceable.

Presentation recording: not published.

## Limits and conclusion

The comparison highlights local features, representation and recipe trade-offs on a small grayscale task. Final validation is previously seen and checkpoint-selected; three seeds describe training variation on one split. Discovery reuse leaves selection bias, and fresh matched final C0 anchors are absent. There is no new blind official test, global-optimum claim or proof that one architecture family is universally better.

## Authors and AI disclosure

AI assisted LaTeX writing and pipeline development. [AI_USAGE.md](AI_USAGE.md).
'''
    _, extension_markdown = garmentlab_excerpt()
    if extension_markdown:
        readme = readme.replace("## Read the project\n", extension_markdown + "## Read the project\n", 1)
        assignment = assignment.replace("## Limits and conclusion\n", extension_markdown + "## Limits and conclusion\n", 1)
    return readme, assignment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check generated pages against current publication data without writing")
    args = parser.parse_args(argv)
    data = json.loads(DATA.read_text(encoding="utf-8"))
    if data['schema_version'] != 'a1_final_publication_v1' or data['study']['task_count'] != 61:
        raise ValueError('Unexpected verified publication schema or task count')
    index, study = build_pages(data)
    readme, assignment = markdown_pages(data)
    if (ROOT / "confirmatory-results.html").is_file():
        followup = '''## Matched follow-up and prospective image audit

The follow-up completes **30 matched 60-epoch fits**, using seeds 1009, 2027 and 3037 with the original C0 and frozen OPT3-selected recipes. Three checkpoint policies yield **90 evaluations**. The predeclared primary CNN epoch-60 difference is **+0.000912 F1**, with paired sample SD **0.003511** and a positive effect in only one of three pairs. This does not establish a stable CNN recipe advantage. MLP and Transformer improve in all three secondary fixed-horizon pairs. All Fashion-MNIST follow-up results remain on previously inspected validation images.

A prospectively locked RGB audit evaluates six fixed models on **2,766 project-unused crops from 1,826 source/duplicate components**. C0/C4 mean F1 is **0.774494/0.781540**. The paired difference **+0.007046** has component-bootstrap interval **[-0.000360, +0.015124]**, leaving the positive population effect unresolved. The preset C4 policy accepts 1,506 crops, with 1,445 correct: **95.95% accepted accuracy at 54.45% coverage**. This same-source license shift is separate from the earlier assessment and from Fashion-MNIST.

[Follow-up results](confirmatory-results.html) · [Full methods and tables](docs/a1_extension_research/CONFIRMATORY_RESULTS_20261030.md) · [Locked protocol](docs/a1_extension_research/CONFIRMATORY_PROTOCOL_20261029.md)

'''
        readme = readme.replace('## Final exploratory validation', followup + '## Earlier optimization study')
        readme = readme.replace('No matched final C0 anchors isolate causal gains over historical CE-selected models.', 'The 61-fit phase did not include matched final C0 anchors. The later 30-fit follow-up supplies those controls on the reused validation split.')
        readme = readme.replace('Presentation recording: not published.\n\n', '')
        verify = '''## Verify the completed follow-up

```bash
python tools/verify_confirmatory_evidence.py
```

The [recorded A100 notebook](notebooks/CO3133_Confirmatory_A100.ipynb) uses the [immutable source archive](assets/frozen_sources/CO3133_Confirmatory_Source_20261029.zip). Its code preserves the executed protocol.

This standard-library command verifies the retained payload hashes and all 90 checkpoint identities, then reconstructs the 90 Fashion-MNIST scores from 540,000 prediction rows. It checks all 1,800 history rows and checkpoint decisions without loading model objects or the official test. The compact publication omits optimizer/RNG recovery states; exact resume uses the original complete run directory. [Independent records](output/confirmatory_return_20261030/independent/INDEPENDENT_STATISTICS.json) document the separate official-training-label check and [CPU portability replay](output/confirmatory_return_20261030/independent/CHECKPOINT_CPU_REPLAY.json). CPU/PyTorch 2.10 changes 11 of 540,000 cell-image predicted labels relative to the A100/PyTorch 2.11 export; original A100 measurements remain the reported results.

'''
        software = '''## CPU software checks

```bash
python -m pip install -r requirements-public-tests.txt
python -m pytest tests -q
```

The CPU profile covers the architecture, study controllers, API and release checks. Historical notebook tests use fake controllers and preserve the archived executable cells. Their required notebooks and the frozen original baseline reference are included. Tests requiring optional raw training data or a separate reference-service release declare their skips explicitly.

The release gate in `tools/verify_public_package.py` checks a supplied publication ZIP against its manifest, frozen source and test fixtures. Its `--software` option runs the complete CPU suite from a fresh extraction and verifies that imports resolve to that extracted source. These checks do not execute the measured A100 study or create a new model-performance result.

'''
        policy_note='''## Selective prediction by garment class

The pooled accepted accuracy of 95.95% conceals class-dependent operating risk. With the fixed C4 seed314 policy, true shirt/blouse crops have **18.13% coverage and 66.67% accepted accuracy** (22/33 accepted crops correct). In the separate grouping by emitted labels, accepted jacket outputs have **86.36% precision** (76/88). All eleven accepted true-shirt errors are labeled as other categories, so changing only the predicted-shirt threshold would not address those cases. Both tables concern one fixed model, use different denominators and remain post-hoc diagnostics. No weights, temperature or threshold changed.

[Six-class policy tables](confirmatory-results.html#policy) · [Counts, uncertainty and definitions](output/policy_by_class_20261030/POLICY_BY_CLASS.json) · [Accepted confusion matrix](output/policy_by_class_20261030/accepted_confusion.csv)

Recompute the diagnostic from the saved labels and logits:

```bash
python tools/analyze_policy_by_class.py
```

'''
        readme = readme.replace('## Course assignment index', verify + software + policy_note + '## Course assignment index')
        assignment = assignment.replace('Presentation recording: not published.\n\n', '')
        assignment += '\n\n## Matched follow-up\n\n[Completed matched recipe/checkpoint study and prospective RGB audit](confirmatory-results.html). The primary CNN comparison is inconclusive across the three seed pairs; all prior benchmark tables retain their historical scope.\n'
        excerpt = '<section class="section container"><h2>Matched follow-up and prospective RGB audit</h2><p>Thirty paired training fits supply 90 checkpoint evaluations. The primary CNN epoch-60 recipe difference is +0.000912 Macro-F1, positive in one of three seed pairs. A separate locked RGB cohort contains 2,766 crops in 1,826 source/duplicate components; its C4 minus C0 interval includes zero.</p><p><a href="confirmatory-results.html">Read the completed follow-up and its evidence</a></p></section>'
        index = index.replace('</section>', '</section>' + excerpt, 1)
        study = study.replace('</section>', '</section>' + excerpt, 1)
        for old,new in [("No fresh C0 anchor is trained", "The earlier OPT3 final phase did not train a fresh C0 anchor"), ("Presentation recording: not published.", "")]:
            index=index.replace(old,new);study=study.replace(old,new)
    readme = readme.replace("## Team", "This A1 project is stored in `assignment1/a1_m2/`. Run the commands below from that directory.\n\n## Team", 1)
    readme = readme.replace("](assignment2.md)", "](../../assignment2/a2_m1/assignment2.md)")
    readme = readme.replace("](assignment3.md)", "](../../assignment3/a3_m1/assignment3.md)")
    outputs = {"index.html":index,"assignment1-final.html":study,"README.md":readme,"assignment1-final.md":assignment}
    extension_page = ROOT / "garmentlab-results.html"
    if extension_page.is_file() and garmentlab_v2_summary():
        existing_page = extension_page.read_text(encoding="utf-8")
        # Normalize the preserved heading before the idempotent insertion point.
        existing_page = existing_page.replace('<h2>Historical v1 study: four data roles</h2>', '<h2>Four data roles.</h2>', 1)
        outputs["garmentlab-results.html"] = add_garmentlab_v2_section(existing_page)
    for relative, content in outputs.items():
        target = ROOT / relative
        if args.check:
            if not target.is_file() or target.read_text(encoding="utf-8") != content:
                raise ValueError('Generated page is stale: ' + relative)
        else:
            target.write_text(content,encoding="utf-8")
    print(('CHECKED' if args.check else 'BUILT') + ': ' + ', '.join(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
