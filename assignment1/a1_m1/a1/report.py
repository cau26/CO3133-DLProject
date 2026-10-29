"""Generate a partial draft from actual saved metrics; never invent results."""
from pathlib import Path

from .common import load_json, save_csv


def create_report(output_dir):
    out = Path(output_dir)
    results = [load_json(out / model / "metrics.json") for model in ["linear", "mlp"]]
    if results[0]["data"]["split_sha256"] != results[1]["data"]["split_sha256"]:
        raise ValueError("The models used different splits: rerun with one shared configuration.")
    if results[0]["config"] != results[1]["config"]:
        raise ValueError("The models used different configurations: document a controlled comparison first.")
    if results[0]["source"]["source_sha256"] != results[1]["source"]["source_sha256"]:
        raise ValueError("The source files changed between model runs. Rerun both models together.")
    cfg, data = results[0]["config"], results[0]["data"]
    rows = [{"model": r["model"], "val_accuracy": r["accuracy"], "val_macro_f1": r["macro_f1"],
             "parameters": r["parameters"], "best_epoch": r["best_epoch"],
             "train_seconds": r["train_seconds"],
             "inference_ms_per_image": r["inference"]["per_image_ms_amortized"]} for r in results]
    save_csv(out / "comparison.csv", rows)
    table = ["| Model | Val accuracy | Val macro-F1 | Parameters | Best epoch | Train (s) | Inference (ms/image) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    table += [f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['parameters']:,} | "
              f"{r['best_epoch']} | {r['train_seconds']:.2f} | "
              f"{r['inference']['per_image_ms_amortized']:.6f} |" for r in results]
    comparison = "\n".join(table)
    (out / "comparison.md").write_text(comparison + "\n", encoding="utf-8")
    delta = (results[1]["accuracy"] - results[0]["accuracy"]) * 100
    parts = [
        "# Assignment 1 - M1 Draft\n",
        f"**Group:** {cfg['group']}  \n**Course:** CO3133, Semester-261  \n**Instructor:** Lê Thành Sách\n",
        "**Members:** " + "; ".join(cfg.get("members", [])) + "\n",
        "> This summary is generated from saved experiment results. Review the figures and expand the analysis before submission. All metrics below are validation results; the test set has not been evaluated.\n",
        "## Part 1 - Problem and Data Description\n",
        "The task is to classify a grayscale fashion image into one of 10 classes. "
        "Each input has shape 1 x 28 x 28, and each model returns 10 logits. "
        "This is single-label, multiclass classification.\n",
        "We use [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), "
        "published by Zalando Research under the MIT license. Images and labels are downloaded "
        "as gzip-compressed IDX files. Torchvision checks their MD5 checksums. "
        "The download source and expected checksums identify the dataset files used in this run.\n",
        f"The split contains {data['train']:,} training, {data['validation']:,} validation and {data['test']:,} test images. "
        f"We stratify the 60,000 official training images by class with seed {cfg['seed']} and reserve the official test set. "
        "Training and validation indices are disjoint and saved in `split_indices.npz`. "
        "Image-content duplicates have not been checked; disjoint indices do not rule them out.\n",
        "![Class distribution](eda/class_distribution.png)\n\n![Sample images](eda/samples.png)\n",
        "Exact counts are saved in [class_counts.csv](eda/class_counts.csv). "
        "The classes have equal sample counts in each split. Images have low resolution and one channel. "
        "Results on this benchmark do not establish performance on real product photographs.\n",
        "`ToTensor()` converts uint8 values in [0, 255] to float32 values in [0, 1]. "
        "These baselines use no data augmentation or additional mean/std normalization. "
        "We use torchvision's Dataset, Subset for splitting and DataLoader for batching.\n",
        "![Training batch after preprocessing](eda/batch_preview.png)\n",
        "## Part 2 - Methodology\n",
        "Pipeline: images and labels → ToTensor → split and DataLoader → Linear or MLP → "
        "CrossEntropyLoss → Adam updates → validation → checkpoint selection → argmax and evaluation.\n",
        f"- Linear: Flatten → Linear(784,10).\n- MLP: Flatten → Linear(784,{cfg['hidden_size']}) → ReLU → Linear({cfg['hidden_size']},10).\n",
        "Linear applies one affine transformation to flattened pixels. "
        "MLP adds a hidden layer and ReLU to learn nonlinear relationships. "
        "Neither model uses convolution to represent local spatial structure. "
        "Both return logits directly to CrossEntropyLoss, without applying Softmax first.\n",
        f"Adam, learning rate {cfg['learning_rate']}, batch size {cfg['batch_size']}, "
        f"{cfg['epochs']} epochs, seed {cfg['seed']}, {cfg['cpu_threads']} CPU threads, "
        f"device {cfg['device']}. We use no scheduler, dropout, weight decay, early stopping or mixed precision. "
        "The checkpoint with the lowest validation loss is saved; ties keep the earlier checkpoint.\n",
        "The experiment tests whether a nonlinear hidden layer improves classification. "
        "The architecture changes, while data, split, preprocessing, loss, optimizer, learning rate, "
        "batch size and epoch budget stay fixed. Parameter counts are not matched, so any improvement "
        "cannot be attributed to nonlinearity alone. This is a baseline configuration, without hyperparameter search.\n",
        "The repository implements the models, training and validation loop, checkpoint selection, EDA and result summary. "
        "PyTorch supplies layers, autograd, optimizers and loss functions; torchvision supplies the data and ToTensor; "
        "scikit-learn handles the split and metrics; matplotlib creates the plots.\n",
        "## Part 3 - Implementation Results\n", comparison + "\n",
        f"MLP minus Linear: {delta:+.2f} percentage points in validation accuracy for this run. "
        "Macro-F1, parameter count and timing provide the rest of the comparison. "
        "This is one seed; we do not report repeated-run mean/std or statistical significance.\n",
        "Training time includes batch loading, forward and backward passes, updates and metric collection, "
        "but excludes validation. Inference timing covers only model forward passes with inputs already on the device: "
        f"batch size {cfg['batch_size']}, 10 warm-up iterations and 50 timed iterations. "
        "Time per image is the batch time divided by batch size, not single-request or end-to-end latency.\n",
    ]
    for r in results:
        name = r["model"]
        parts += [f"### {name.upper()}\n",
                  f"![Curves]({name}/curves.png)\n\n![Confusion matrix]({name}/confusion_matrix.png)\n\n![Examples]({name}/examples.png)\n",
                  "The most frequent validation confusions are:\n"]
        parts += [f"- {e['true_class']} → {e['predicted_class']}: {e['count']} images "
                  f"({e['rate_within_true_class']:.1%} of the true class).\n" for e in r["top_errors"]]
        parts += [f"Hardware: {r['environment']['cpu']}; device {r['environment']['device']}. "
                  f"Settings, versions, source fingerprint, timings and checkpoint details are in [{name}/metrics.json]({name}/metrics.json).\n"]
    parts += [
        "### Analysis to add to the full report\n",
        "1. Read both learning curves. Does validation stop improving while training loss keeps falling?\n"
        "2. Discuss at least two incorrect predictions. Describe visible details and possible causes.\n"
        "3. Explain the accuracy, macro-F1, speed and parameter-count trade-offs.\n"
        "4. Describe implementation issues that actually occurred and how they were handled.\n",
        "### Limitations and final-submission plan\n",
        "This draft covers Linear and MLP with one seed and one configuration. The test set has not been evaluated. "
        "Next steps are a custom CNN, an LSTM or GRU, and a Transformer, followed by a broader comparison "
        "of representations and inductive biases. Evaluate the test set after making model choices on validation. "
        "Complete the report, slides, video and links required by the handbook.\n",
        "### AI Usage Disclosure\n",
        "The team used AI to develop ideas, review source code and improve the report. "
        "Tools, scope and verification are documented in the repository's AI_USAGE.md.\n",
        "### References\n",
        "- Course Project Handbook CO3133, revision 14 September 2026, Sections 3, 5, 7.4 and Part II.\n"
        "- [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).\n"
        "- [PyTorch Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).\n",
    ]
    (out / "draft_report.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"Draft and comparison saved: {out / 'draft_report.md'}", flush=True)
