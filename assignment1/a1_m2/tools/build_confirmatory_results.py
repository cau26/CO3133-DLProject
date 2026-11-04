"""Derive the public matched-study account from completed, saved evidence.

This publication command performs no training, inference, calibration or model
selection. It preserves the distinction between reused validation, fixed model
seed variation and a prospectively locked RGB image cohort.
"""
from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "output/confirmatory_return_20261030"
FAMILIES = ("linear", "mlp", "cnn", "gru", "transformer")
NAMES = {"linear": "Linear", "mlp": "MLP", "cnn": "CNN", "gru": "GRU", "transformer": "Transformer"}
MARKDOWN = "docs/a1_extension_research/CONFIRMATORY_RESULTS_20261030.md"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def md_table(headings, rows):
    return "\n".join(["| " + " | ".join(headings) + " |", "|" + "---|" * len(headings),
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def html_table(headings, rows, caption):
    esc = lambda s: html.escape(str(s))
    head = "".join(f'<th scope="col">{esc(v)}</th>' for v in headings)
    body = []
    for row in rows:
        cells = f'<th scope="row">{esc(row[0])}</th>' + "".join(f"<td>{esc(v)}</td>" for v in row[1:])
        body.append("<tr>" + cells + "</tr>")
    return (f'<div class="table-wrap" role="region" aria-label="{esc(caption)}" tabindex="0">'
            f"<table><caption>{esc(caption)}</caption><thead><tr>{head}</tr></thead>"
            f'<tbody>{"".join(body)}</tbody></table></div>')


def build():
    summary = read(BASE + "/fashion_confirmation/analysis/summary.json")
    status = read(BASE + "/RESULT_STATUS.json")
    manifest = read(BASE + "/fashion_confirmation/confirmation_manifest.json")
    rgb = read(BASE + "/rgb_prospective/PROSPECTIVE_RESULTS.json")
    cpu_replay = read(BASE + "/independent/CHECKPOINT_CPU_REPLAY.json")
    if status["status"] != "COMPLETE" or status["fashion"]["completed_trajectories"] != 30:
        raise ValueError("A completed 30-trajectory return is required for this publication.")
    if status["fashion"]["completed_checkpoint_cells"] != 90 or summary["official_test_evaluated"]:
        raise ValueError("Unexpected checkpoint count or official-test use.")
    if rgb["smoke_only"] or rgb["cohort"]["crops"] != 2766 or rgb["cohort"]["components"] != 1826:
        raise ValueError("Unexpected prospective cohort identity.")
    cells = {(r["family"], r["arm"], r["policy"]): r for r in summary["cells"]}
    contrasts = {(r["family"], r["contrast"]): r for r in summary["paired_contrasts"]}
    if len(cells) != 30 or any(r["n_seeds"] != 3 for r in cells.values()):
        raise ValueError("Expected two recipe arms × three selectors × five families, each over three seeds.")

    mean_sd = lambda row: f'{row["macro_f1_mean"]:.5f} ± {row["macro_f1_sample_sd"]:.5f}'
    headings = ["Family / recipe", "Minimum CE", "Maximum F1", "Fixed epoch 60"]
    rows = []
    for family in FAMILIES:
        for arm in ("C0", "selected"):
            candidate = manifest["protocol"]["recipes"][family][arm]["candidate_id"]
            label = NAMES[family] + " / " + candidate + (" (selected)" if arm == "selected" else " (anchor)")
            rows.append([label, *[mean_sd(cells[(family, arm, policy)]) for policy in ("ce", "f1", "final")]])
    delta_headings = ["Family", "Epoch-60 recipe Δ", "Recipe Δ at CE", "Recipe Δ at F1", "Epoch-60 positive pairs"]
    delta_rows = []
    for family in FAMILIES:
        fixed = contrasts[(family, "recipe_fixed_final_primary")]
        delta_rows.append([NAMES[family], f'{fixed["macro_f1_delta_mean"]:+.5f}',
                           f'{contrasts[(family,"recipe_at_ce")]["macro_f1_delta_mean"]:+.5f}',
                           f'{contrasts[(family,"recipe_at_f1")]["macro_f1_delta_mean"]:+.5f}',
                           f'{sum(x > 0 for x in fixed["macro_f1_delta_per_seed"])}/3'])
    cnn = contrasts[("cnn", "recipe_fixed_final_primary")]
    cnn_rows = [[str(seed), f"{delta:+.6f}"] for seed, delta in zip(cnn["seeds"], cnn["macro_f1_delta_per_seed"])]

    rgb_headings = ["Fixed model recipe", "Accuracy (%)", "Macro-F1"]
    rgb_rows = []
    for arm in ("C0", "C4"):
        models = [m for m in rgb["models"] if m["candidate_id"] == arm]
        values = [m["raw"]["macro_f1"] for m in models]
        rgb_rows.append([arm, f'{statistics.mean(m["raw"]["accuracy"] for m in models)*100:.3f}',
                         f'{statistics.mean(values):.5f} ± {statistics.stdev(values):.5f}'])
    class_headings = ["Class", "Crops", "Components containing class", "C0 mean class F1", "C4 mean class F1"]
    class_rows = []
    for index, (label, support) in enumerate(rgb["cohort"]["class_support"].items()):
        means = [statistics.mean(m["raw"]["per_class"][index]["f1"] for m in rgb["models"] if m["candidate_id"] == arm)
                 for arm in ("C0", "C4")]
        class_rows.append([label.replace("_", " "), support["crops"], support["components"], *[f"{x:.5f}" for x in means]])
    primary = rgb["primary"]
    lo, hi = primary["uncertainty"]["paired_mean_macro_f1_delta_interval"]
    rgb_delta = primary["paired_mean_macro_f1_delta"]
    fixed_model = next(m for m in rgb["models"] if m["candidate_id"] == "C4" and m["seed"] == 314)
    policy = fixed_model["frozen_policy"]
    core_hours = status["fashion"]["clock"]["active_run_seconds"] / 3600
    versions = manifest["environment"]["versions"]

    intro = ("A matched study separates changes to the training recipe from changes to checkpoint selection. "
             "The predeclared primary comparison, CNN C5 versus C0 at epoch 60, has a small positive mean "
             "difference and opposite signs across the three training seeds. The evidence does not establish "
             "a consistent CNN recipe improvement. A separate, prospectively locked RGB audit adds image "
             "evidence outside recorded project exposure, with a positive C4 point estimate whose grouped "
             "uncertainty interval includes zero.")
    design = ("Five Fashion-MNIST families are each trained with the original C0 anchor and the recipe selected "
              "by OPT3, using fresh seeds 1009, 2027 and 3037. Each trajectory runs 60 epochs on the same "
              "54,000 training images. Its 6,000-image validation partition has been used previously in the "
              "project. Minimum-validation-CE, maximum-validation-Macro-F1 and fixed-epoch-60 checkpoints "
              "are read from each trajectory. The completed run therefore comprises 30 fits, 1,800 training "
              "epochs and 90 checkpoint cells; the cells are correlated observations from the same fits.")
    primary_text = (f"CNN C0 records epoch-60 Macro-F1 {cells[('cnn','C0','final')]['macro_f1_mean']:.5f}, "
                    f"compared with {cells[('cnn','selected','final')]['macro_f1_mean']:.5f} for C5. "
                    f"The paired mean difference is {cnn['macro_f1_delta_mean']:+.6f}, with sample SD "
                    f"{cnn['macro_f1_delta_sample_sd']:.6f} across three pairs. Only one pair favors C5. "
                    "This does not support a claim of a stable recipe gain. C5 remains the highest observed "
                    "family mean under maximum-F1 selection in this run, but that is a different, secondary "
                    "comparison and does not establish an architecture-wide or unseen-data advantage.")
    selector_text = ("Recipe effects depend on the checkpoint rule. Linear, GRU and Transformer have negative "
                     "mean recipe differences at the CE-selected checkpoint despite positive mean differences "
                     "at fixed epoch 60. For Transformer, the mean changes from −0.00038 at minimum CE to "
                     "+0.00946 at epoch 60; the selected schedule may preserve late-training performance without "
                     "improving the best CE checkpoint. MLP shows positive recipe differences under all three "
                     "rules on these seeds. These observations concern complete recipe bundles; they do not "
                     "identify separate causal effects of learning rate, decay or dropout.")
    selection_caution = ("Maximum validation F1 is at least as high as F1 at any other observed epoch by "
                         "construction. Its advantage on the same validation set cannot demonstrate better "
                         "generalization. For selected CNN, changing from minimum CE to maximum F1 adds "
                         "+0.00186 mean validation F1, while mean CE rises from 0.19376 to 0.19817. "
                         "Checkpoint choice trades metrics; it is not an independent improvement experiment.")
    old_comparison = ("The earlier OPT3 CNN mean, 0.93308 on seeds 42–44, and the new C5 maximum-F1 mean, "
                      "0.93126 on seeds 1009/2027/3037, describe different training seeds and recorded runs. "
                      "Their difference is not a software regression diagnosis, a replacement test score or "
                      "evidence that the earlier result was invalid. The matched C0 versus selected comparisons "
                      "within this run are the appropriate recipe evidence.")
    rgb_design = ("The RGB protocol, source hashes, six existing C0/C4 checkpoints and cohort rule were frozen "
                  "before new-image preparation and inference. A fixed hash rank samples source groups from "
                  "Fashionpedia license IDs 3/5; recorded image/URL exposure and source/duplicate components "
                  "connected to historical images are excluded. After annotation filtering, the audit contains "
                  "2,766 garment crops in 1,826 source/duplicate components. Each component may contain several "
                  "classes, so class-component counts do not sum to the total. The six fixed models use "
                  "training seeds 314, 2718 and 5772; their predictions are averaged as metrics, not combined "
                  "into an ensemble.")
    rgb_result = (f"C4−C0 mean Macro-F1 is {rgb_delta:+.6f}. A paired source/duplicate-component percentile "
                  f"bootstrap with 2,000 replicates gives the 95% interval [{lo:+.6f}, {hi:+.6f}]. "
                  "The interval includes zero, so the audit does not establish a positive C4 effect. It is "
                  "conditional on these six fixed models and the component-sampling procedure, not an "
                  "interval for the population of possible training seeds. The earlier 327-crop assessment "
                  "gave the opposite point-estimate direction; retaining both results exposes sensitivity "
                  "to cohort composition instead of replacing an unfavorable table.")
    rgb_error = ("Shirt/blouse remains the weakest category: mean C4 class F1 is 0.56594, compared with "
                 "0.91846 for pants. Jacket has the largest observed class-F1 increase, +0.01841. This is "
                 "descriptive class analysis across the fixed models, with no class-level multiplicity "
                 "correction. More crops improve the evidence base but do not remove label ambiguity, "
                 "class imbalance or source bias.")
    policy_text = (f"For the preselected C4 seed314 model, the existing temperature and threshold 0.88 are "
                   f"reused without fitting on the new cohort. The policy accepts {policy['accepted_count']:,} "
                   f"of {policy['samples']:,} crops ({policy['coverage']*100:.2f}% coverage); "
                   f"{policy['accepted_correct']:,} are correct ({policy['accepted_accuracy']*100:.2f}% accepted "
                   "accuracy). These are selective, descriptive measurements, not accuracy on every input "
                   "or a transferred 95% risk guarantee. The calibration population differs from this cohort.")
    detail = read('output/policy_by_class_20261030/POLICY_BY_CLASS.json')
    if detail['training_seed'] != 314 or detail['review_threshold'] != .88 or detail['overall']['accepted_count'] != 1506:
        raise ValueError('Unexpected fixed-model policy diagnostic.')
    def policy_rows(kind):
        return [[r['label'].replace('_',' '), f"{r['accepted_count']}/{r['support']}",
                 f"{r['accepted_correct']}/{r['accepted_count']}", str(r['review_count']),
                 f"{100*r['coverage']:.2f}", f"{100*r['accepted_accuracy']:.2f}"]
                for r in detail[kind]]
    true_headers=['True category','Accepted / total','Correct / accepted','Review','Coverage (%)','Accepted accuracy (%)']
    pred_headers=['Predicted output','Accepted / predictions','Correct / accepted','Review','Coverage (%)','Accepted precision (%)']
    true_rows,pred_rows=policy_rows('true_class'),policy_rows('predicted_class')
    policy_intro=('This post-hoc diagnostic concerns C4 seed314 alone, using the unchanged temperature '
                  '5.345757234366385 and acceptance rule max softmax(logits / temperature) ≥ 0.88. '
                  'It does not average policies over training seeds or fit class-specific thresholds.')
    true_discussion=('The true-class table conditions on the annotated garment category. Shirt/blouse '
                     'has 33 accepted crops among 182, with 22 correct: 18.13% coverage and 66.67% accepted '
                     'accuracy. It contributes 11 of the 61 accepted errors despite only 2.19% of the accepted '
                     'set. These labels are available for offline auditing and cannot determine a routing rule '
                     'for a new unlabeled crop.')
    pred_discussion=('The predicted-class table conditions on the emitted label. Accepted jacket outputs '
                     'have precision 76/88 = 86.36%, while accepted shirt/blouse outputs have precision '
                     '22/24 = 91.67%. All 11 accepted errors on true shirt/blouse crops are instead labeled '
                     'jacket (five), top/T-shirt/sweatshirt (four) or dress (two). Tightening only the predicted '
                     'shirt/blouse threshold would not directly address those errors. The two table denominators '
                     'answer different questions and must not be interchanged.')
    policy_scope=('Review contains 434 of the 495 full-cohort classification errors, but also 826 predictions '
                  'that were already correct. Human review outcomes and costs were not measured. The JSON '
                  'retains 2,000-replicate pointwise component-bootstrap intervals, conditional on this fixed '
                  'model; they do not provide simultaneous classwise or future-population certification. '
                  'The true-label grouping is not a direct test of probability calibration. No policy is '
                  'modified using these inspected outcomes. A class-aware policy needs separate calibration '
                  'and an untouched assessment cohort.')
    policy_md=['## Review policy by class',policy_intro,'### Grouped by true category',
               md_table(true_headers,true_rows),true_discussion,'### Grouped by predicted output',
               md_table(pred_headers,pred_rows),pred_discussion,policy_scope]
    scope_text = ("The cohort is unused relative to the recorded project exposure and drawn from the same "
                  "Fashionpedia source under a license-cohort shift. It is neither an external collection nor "
                  "an IID benchmark. Previously unrecorded image views and ImageNet pretraining exposure "
                  "cannot be excluded. Once these outcomes are inspected, further RGB tuning needs another "
                  "untouched audit population. The Fashion-MNIST official test partition was not loaded; "
                  "fresh training seeds cannot make the old validation images new.")
    runtime_text = (f"The Fashion-MNIST controller records {core_hours:.3f} active hours on NVIDIA "
                    f"A100-SXM4-40GB, Python 3.13, PyTorch {versions['torch']}, torchvision "
                    f"{versions['torchvision']}, CUDA 13.0, NumPy {versions['numpy']} and scikit-learn "
                    f"{versions['scikit-learn']}. Setup and smoke checks are excluded from that clock. "
                    "The prospective RGB inference was completed separately on CPU with PyTorch 2.10.0 "
                    "and torchvision 0.25.0; its frozen evidence without source photographs was verified and reused by the "
                    "A100 bundle. It was not a new RGB training or calibration experiment on Colab.")
    replay_text = (f"A separate CPU replay loads all {cpu_replay['checkpoints']} locked checkpoints without "
                   f"changing their weights. Compared with the original A100 prediction files, PyTorch 2.10 CPU "
                   f"produces {cpu_replay['total_label_differences']} label differences across "
                   f"{cpu_replay['prediction_comparisons']:,} checkpoint–image comparisons; these are not "
                   f"necessarily distinct images. The largest absolute per-cell mean NLL difference is "
                   f"{cpu_replay['maximum_mean_nll_difference']:.5g}, and the largest per-image NLL difference "
                   f"is {cpu_replay['maximum_per_image_nll_difference']:.5f}. The original A100 measurements "
                   "are retained. This check documents numerical variation across hardware and software "
                   "rather than claiming exact identity; the mixed-sign primary CNN conclusion is unchanged.")
    conclusion = ("The additional study strengthens attribution and reproducibility rather than establishing "
                  "universal recipe superiority. It fills the missing matched-anchor comparison, reveals "
                  "checkpoint-dependent conclusions and adds a prospectively locked image audit. The CNN "
                  "primary contrast remains inconclusive across these seeds, and the RGB effect remains "
                  "uncertain under grouped sampling. Independent images, additional training seeds and "
                  "broader operational validation are still required for stronger deployment claims.")
    evidence = [
        (BASE + "/RESULT_STATUS.json", "Run completion and scope"),
        (BASE + "/fashion_confirmation/confirmation_manifest.json", "Recipes, seeds, environment and source identities"),
        (BASE + "/fashion_confirmation/analysis/summary.json", "All checkpoint aggregates and paired contrasts"),
        (BASE + "/fashion_confirmation/EVALUATION_LOCK.json", "Ninety locked checkpoint identities"),
        (BASE + "/rgb_prospective/PROSPECTIVE_FREEZE.json", "Prospective RGB freeze"),
        (BASE + "/rgb_prospective/PROSPECTIVE_RESULTS.json", "RGB logits-derived results and component intervals"),
        (BASE + "/independent/CHECKPOINT_CPU_REPLAY.json", "CPU checkpoint replay and numerical differences"),
        (BASE + "/independent/INDEPENDENT_STATISTICS.json", "Independent prediction and trajectory statistics"),
        ("output/policy_by_class_20261030/POLICY_BY_CLASS.json", "Class-conditional policy counts and pointwise uncertainty"),
        ("output/policy_by_class_20261030/accepted_confusion.csv", "Confusion counts among accepted predictions"),
        ("docs/a1_extension_research/CONFIRMATORY_PROTOCOL_20261029.md", "Predeclared protocol"),
        ("AI_USAGE.md", "AI usage"),
    ]
    md = ["# Matched training and prospective image evaluation", "CO3133 · Assignment 1 · Group G-M10", intro,
          "## Questions and experimental design", design,
          "CNN selected−C0 at fixed epoch 60 is the predeclared primary contrast. Other families, CE/F1 selectors and their interactions are secondary exploratory analyses. Mean ± SD below is the sample standard deviation across three training seeds on one fixed split; it is not a confidence interval over unseen images.",
          "## All recipe and checkpoint cells", md_table(headings, rows),
          "## Primary CNN result", primary_text, md_table(["Paired seed", "C5−C0 epoch-60 Macro-F1"], cnn_rows),
          "## Recipe and checkpoint attribution", md_table(delta_headings, delta_rows), selector_text, selection_caution, old_comparison,
          "## Prospective RGB image audit", rgb_design, md_table(rgb_headings, rgb_rows), rgb_result,
          "## Class errors and selective predictions", md_table(class_headings, class_rows), rgb_error, policy_text,
          *policy_md,
          "## Scope, runtime and reproducibility", scope_text, runtime_text, replay_text,
          "AI assisted LaTeX writing and pipeline development.",
          "## Interpretation", conclusion, "## Evidence", *[f"- [{label}](../../{path})" for path,label in evidence]]

    def para(s):
        return "<p>" + html.escape(s) + "</p>"

    content = (
        '<section id="design"><p class="eyebrow">A matched follow-up</p><h2>Recipe changes and checkpoint choices.</h2>'
        + para(design)
        + para("CNN selected−C0 at fixed epoch 60 is the predeclared primary contrast. Other family, selector and interaction comparisons are secondary exploratory analyses.")
        + html_table(headings, rows, "Validation Macro-F1 · mean ± sample SD across three training seeds")
        + para("The validation images were previously used. Across-seed SD is not a population confidence interval, and 90 checkpoint cells are not 90 independent experiments.") + "</section>"
        + '<section id="cnn"><p class="eyebrow">Predeclared primary contrast</p><h2>CNN: a modest, inconsistent difference.</h2>'
        + para(primary_text) + html_table(["Paired seed", "C5−C0 epoch-60 Macro-F1"], cnn_rows, "Paired seed effects on the fixed validation split") + "</section>"
        + '<section id="attribution"><h2>The checkpoint rule changes the conclusion.</h2>'
        + html_table(delta_headings, delta_rows, "Selected recipe minus C0 · Macro-F1 differences")
        + para(selector_text) + '<div class="note">' + para(selection_caution) + "</div>" + para(old_comparison) + "</section>"
        + '<section id="rgb"><p class="eyebrow">A separate image population</p><h2>2,766 RGB crops. Six fixed models.</h2>'
        + para(rgb_design) + html_table(rgb_headings, rgb_rows, "Prospective cohort results · no ensemble and no refitting")
        + para(rgb_result) + "</section>"
        + '<section id="errors"><h2>Class errors and selective coverage.</h2>'
        + html_table(class_headings, class_rows, "Class support and mean class F1 across three fixed models per recipe")
        + para(rgb_error) + para(policy_text) + "</section>"
        + '<section id="scope"><h2>What the evidence can support.</h2>' + para(scope_text) + para(runtime_text) + para(replay_text)
        + para(conclusion)
        + '<p class="muted">AI assisted LaTeX writing and pipeline development. <a href="AI_USAGE.md">AI usage →</a></p></section>'
        + '<section id="evidence"><h2>Inspect the evidence.</h2><ul class="resource-list">'
        + "".join(f'<li><a href="{html.escape(path)}">{html.escape(label)} →</a></li>' for path,label in evidence)
        + f'<li><a href="{MARKDOWN}">Scientific analysis in Markdown →</a></li></ul></section>'
    )
    policy_html=('<section id="policy"><h2>Review policy by class</h2>'+para(policy_intro)
        +html_table(true_headers,true_rows,'Frozen C4 seed314 policy, grouped by true category')+para(true_discussion)
        +html_table(pred_headers,pred_rows,'Same policy, grouped by predicted output')+para(pred_discussion)+para(policy_scope)+'</section>')
    assert '<section id="scope">' in content
    content=content.replace('<section id="scope">',policy_html+'<section id="scope">',1)
    page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Matched training and prospective evaluation · G-M10</title>
<meta name="description" content="A 30-fit matched Fashion-MNIST study and a prospectively locked RGB audit, with checkpoint-dependent findings and explicit uncertainty.">
<meta name="theme-color" content="#243b45"><style>
:root{color-scheme:light;--ink:#18343d;--muted:#50636a;--paper:#faf9f5;--line:#d8dfe0;--accent:#216768}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.7 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--accent);text-underline-offset:.18em}a:hover{color:#123f40}.container{width:min(1150px,calc(100% - 48px));margin:auto}
.skip{position:absolute;left:-9999px}.skip:focus{left:16px;top:16px;padding:12px;background:white;z-index:9}
header{border-bottom:1px solid var(--line);background:#fff}nav{display:flex;justify-content:space-between;align-items:center;gap:24px;padding:19px 0}nav div{display:flex;gap:24px;flex-wrap:wrap}.brand{font-weight:750;text-decoration:none;letter-spacing:.06em}
.hero{padding:72px 0 48px;border-bottom:1px solid var(--line)}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-weight:700;font-size:12px;color:var(--accent)}h1{font-size:clamp(36px,5.5vw,64px);line-height:1.12;font-weight:650;letter-spacing:-.04em;max-width:950px;margin:16px 0 24px}h2{font-size:clamp(25px,3vw,35px);line-height:1.25;letter-spacing:-.025em;font-weight:650;margin:12px 0 22px}.lead{font-size:20px;max-width:980px;color:var(--muted)}
.facts{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin:36px 0 0}.facts div{border-top:2px solid var(--accent);padding-top:12px}.facts dd{margin:4px 0 0}.facts strong{font-size:26px;font-weight:650;display:block}.facts span{font-size:14px;color:var(--muted)}
.layout{display:grid;grid-template-columns:210px minmax(0,1fr);gap:48px}.contents{position:sticky;top:24px;align-self:start;padding:38px 0;font-size:14px}.contents a{display:block;padding:6px 0;text-decoration:none}.contents strong{display:block;margin-bottom:12px}.article{min-width:0}.article section{padding:38px 0;border-bottom:1px solid var(--line);scroll-margin-top:24px}.article p{max-width:840px}.table-wrap{overflow:auto;margin:26px 0;border:1px solid var(--line);background:white;border-radius:7px}.table-wrap:focus{outline:2px solid var(--accent);outline-offset:3px}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.5;font-variant-numeric:tabular-nums}caption{text-align:left;padding:15px;font-size:13px;font-weight:650;color:var(--muted);background:#f0f3f2}th,td{padding:11px 14px;border-top:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead th{font-size:12px;color:var(--muted);white-space:normal}tbody th{font-weight:600}tbody tr:nth-child(even){background:#f8faf9}.note{border-left:3px solid var(--accent);padding:1px 20px;background:#edf3f1;margin:26px 0}.muted{font-size:14px;color:var(--muted)}.resource-list{list-style:none;padding:0}.resource-list li{border-top:1px solid var(--line);padding:12px 0}.resource-list a{text-decoration:none}footer{padding:40px 0;font-size:13px;color:var(--muted)}
@media(max-width:850px){.layout{grid-template-columns:1fr;gap:0}.contents{position:static;padding:25px 0;border-bottom:1px solid var(--line);display:flex;gap:12px;flex-wrap:wrap}.contents strong{flex-basis:100%;margin:0}.contents a{padding:0}.hero{padding-top:44px}.article section{padding:30px 0}}@media(max-width:550px){.container{width:calc(100% - 30px)}body{font-size:16px}nav{align-items:flex-start}nav div{gap:12px;font-size:13px}.facts{gap:12px}.facts strong{font-size:22px}.facts span{font-size:12px}.lead{font-size:18px}th,td{padding:10px}}
</style></head><body><a class="skip" href="#main">Skip to content</a><header><nav class="container" aria-label="Main navigation"><a class="brand" href="index.html">G–M10 · CO3133</a><div><a href="assignment1-final.html">Assignment 1</a><a href="garmentlab-results.html">GarmentLab</a><a href="https://github.com/cau26/CO3133-DLProject">Repository ↗</a></div></nav></header>
<main id="main"><div class="hero"><div class="container"><p class="eyebrow">CO3133 · Assignment 1 · G-M10</p><h1>Separating a better recipe<br>from a better checkpoint.</h1><p class="lead">''' + html.escape(intro) + '''</p><dl class="facts"><div><dt><strong>30 fits</strong></dt><dd><span>Five families · two recipes · three seeds</span></dd></div><div><dt><strong>90 checkpoints</strong></dt><dd><span>CE · Macro-F1 · fixed epoch 60</span></dd></div><div><dt><strong>1,826 components</strong></dt><dd><span>2,766 prospectively locked RGB crops</span></dd></div></dl></div></div>
<div class="container layout"><aside class="contents" aria-label="On this page"><strong>On this page</strong><a href="#design">Matched design</a><a href="#cnn">Primary CNN result</a><a href="#attribution">Checkpoint attribution</a><a href="#rgb">Prospective RGB</a><a href="#errors">Errors and coverage</a><a href="#scope">Scope and limitations</a><a href="#evidence">Evidence</a></aside><article class="article">''' + content + '''</article></div></main><footer class="container">Ho Chi Minh City University of Technology · VNU-HCM<br>Faculty of Computer Science and Engineering · CO3133 Semester 261 · Instructor Lê Thành Sách<br>Trần Gia Lâm · 2352670 &nbsp; / &nbsp; Nguyễn Hữu Cầu · 2352129</footer></body></html>
'''
    page=page.replace('<a href="#scope">Scope and limitations</a>', '<a href="#policy">Policy by class</a><a href="#scope">Scope and limitations</a>')
    return {MARKDOWN: "\n\n".join(md) + "\n", "confirmatory-results.html": page}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify that generated public accounts match recorded evidence.")
    args = parser.parse_args()
    rendered = build()
    for path, text in rendered.items():
        target = ROOT / path
        if args.check:
            if not target.is_file() or target.read_text(encoding="utf-8") != text:
                raise SystemExit("Publication differs from recorded evidence: " + path)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="\n")
    print(("Verified" if args.check else "Built") + ": " + ", ".join(rendered))


if __name__ == "__main__":
    main()
