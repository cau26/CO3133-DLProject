# Assignment 1 - M1 Draft

**Group:** G-M10  
**Course:** CO3133, Semester-261  
**Instructor:** Lê Thành Sách

**Members:** Trần Gia Lâm (2352670); Nguyễn Hữu Cầu (2352129)

> This summary is generated from saved experiment results. Review the figures and expand the analysis before submission. All metrics below are validation results; the test set has not been evaluated.

## Part 1 - Problem and Data Description

The task is to classify a grayscale fashion image into one of 10 classes. Each input has shape 1 x 28 x 28, and each model returns 10 logits. This is single-label, multiclass classification.

We use [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), published by Zalando Research under the MIT license. Images and labels are downloaded as gzip-compressed IDX files. Torchvision checks their MD5 checksums. The download source and expected checksums identify the dataset files used in this run.

The split contains 54,000 training, 6,000 validation and 10,000 test images. We stratify the 60,000 official training images by class with seed 42 and reserve the official test set. Training and validation indices are disjoint and saved in `split_indices.npz`. Image-content duplicates have not been checked; disjoint indices do not rule them out.

![Class distribution](eda/class_distribution.png)

![Sample images](eda/samples.png)

Exact counts are saved in [class_counts.csv](eda/class_counts.csv). The classes have equal sample counts in each split. Images have low resolution and one channel. Results on this benchmark do not establish performance on real product photographs.

`ToTensor()` converts uint8 values in [0, 255] to float32 values in [0, 1]. These baselines use no data augmentation or additional mean/std normalization. We use torchvision's Dataset, Subset for splitting and DataLoader for batching.

![Training batch after preprocessing](eda/batch_preview.png)

## Part 2 - Methodology

Pipeline: images and labels → ToTensor → split and DataLoader → Linear or MLP → CrossEntropyLoss → Adam updates → validation → checkpoint selection → argmax and evaluation.

- Linear: Flatten → Linear(784,10).
- MLP: Flatten → Linear(784,256) → ReLU → Linear(256,10).

Linear applies one affine transformation to flattened pixels. MLP adds a hidden layer and ReLU to learn nonlinear relationships. Neither model uses convolution to represent local spatial structure. Both return logits directly to CrossEntropyLoss, without applying Softmax first.

Adam, learning rate 0.001, batch size 128, 10 epochs, seed 42, 4 CPU threads, device cuda. We use no scheduler, dropout, weight decay, early stopping or mixed precision. The checkpoint with the lowest validation loss is saved; ties keep the earlier checkpoint.

The experiment tests whether a nonlinear hidden layer improves classification. The architecture changes, while data, split, preprocessing, loss, optimizer, learning rate, batch size and epoch budget stay fixed. Parameter counts are not matched, so any improvement cannot be attributed to nonlinearity alone. This is a baseline configuration, without hyperparameter search.

The repository implements the models, training and validation loop, checkpoint selection, EDA and result summary. PyTorch supplies layers, autograd, optimizers and loss functions; torchvision supplies the data and ToTensor; scikit-learn handles the split and metrics; matplotlib creates the plots.

## Part 3 - Implementation Results

| Model | Val accuracy | Val macro-F1 | Parameters | Best epoch | Train (s) | Inference (ms/image) |
|---|---:|---:|---:|---:|---:|---:|
| linear | 0.8662 | 0.8644 | 7,850 | 9 | 64.62 | 0.000312 |
| mlp | 0.8930 | 0.8924 | 203,530 | 9 | 64.04 | 0.001074 |

MLP minus Linear: +2.68 percentage points in validation accuracy for this run. Macro-F1, parameter count and timing provide the rest of the comparison. This is one seed; we do not report repeated-run mean/std or statistical significance.

Training time includes batch loading, forward and backward passes, updates and metric collection, but excludes validation. Inference timing covers only model forward passes with inputs already on the device: batch size 128, 10 warm-up iterations and 50 timed iterations. Time per image is the batch time divided by batch size, not single-request or end-to-end latency.

### LINEAR

![Curves](linear/curves.png)

![Confusion matrix](linear/confusion_matrix.png)

![Examples](linear/examples.png)

The most frequent validation confusions are:

- Shirt → T-shirt/top: 97 images (16.2% of the true class).

- Pullover → Coat: 82 images (13.7% of the true class).

- Shirt → Coat: 73 images (12.2% of the true class).

- Shirt → Pullover: 71 images (11.8% of the true class).

- Coat → Pullover: 58 images (9.7% of the true class).

Hardware: Intel(R) Xeon(R) CPU @ 2.00GHz; device cuda. Settings, versions, source fingerprint, timings and checkpoint details are in [linear/metrics.json](linear/metrics.json).

### MLP

![Curves](mlp/curves.png)

![Confusion matrix](mlp/confusion_matrix.png)

![Examples](mlp/examples.png)

The most frequent validation confusions are:

- Pullover → Coat: 76 images (12.7% of the true class).

- Shirt → T-shirt/top: 69 images (11.5% of the true class).

- T-shirt/top → Shirt: 59 images (9.8% of the true class).

- Shirt → Coat: 46 images (7.7% of the true class).

- Shirt → Pullover: 44 images (7.3% of the true class).

Hardware: Intel(R) Xeon(R) CPU @ 2.00GHz; device cuda. Settings, versions, source fingerprint, timings and checkpoint details are in [mlp/metrics.json](mlp/metrics.json).

### Analysis to add to the full report

1. Read both learning curves. Does validation stop improving while training loss keeps falling?
2. Discuss at least two incorrect predictions. Describe visible details and possible causes.
3. Explain the accuracy, macro-F1, speed and parameter-count trade-offs.
4. Describe implementation issues that actually occurred and how they were handled.

### Limitations and final-submission plan

This draft covers Linear and MLP with one seed and one configuration. The test set has not been evaluated. Next steps are a custom CNN, an LSTM or GRU, and a Transformer, followed by a broader comparison of representations and inductive biases. Evaluate the test set after making model choices on validation. Complete the report, slides, video and links required by the handbook.

### AI Usage Disclosure

The team used AI to develop ideas, review source code and improve the report. Tools, scope and verification are documented in the repository's AI_USAGE.md.

### References

- Course Project Handbook CO3133, revision 14 September 2026, Sections 3, 5, 7.4 and Part II.
- [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).
- [PyTorch Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).
