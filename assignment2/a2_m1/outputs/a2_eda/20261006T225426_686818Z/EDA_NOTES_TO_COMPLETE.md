# Preliminary EDA notes — review before submission

Run ID: 20261006T225426_686818Z
Prepared for review by Nguyễn Hữu Cầu and Trần Gia Lâm.

## Facts computed by this run
- Successfully decoded images: 5011.
- XML object entries (including difficult): 15662.
- Difficult object entries: 3054.
- Valid non-difficult objects: 12608.
- Proposed train / validation images: 4013 / 998.
- Valid non-difficult training objects: 10014.
- Most frequent class by valid non-difficult objects: person (4690).
- Least frequent class by the same definition: diningtable (215).
- Objects per image: mean 3.126, median 2.0, maximum 42.
- Median relative area of valid boxes: 0.066336.
- Annotation issues: 0; perceptual-hash candidates: 3.

## Interpretation to be written by Cầu
### 1. Severe Class Imbalance across Categories
- **Evidence:** `tables/class_counts.csv`, `figures/class_distribution.png`.
- **Observation:** The dataset exhibits severe class imbalance. The dominant category is `person` with 4,690 usable non-difficult instances, whereas the least frequent category is `diningtable` with only 215 usable instances. A random-split baseline was not run, so its effect is not measured here; stratification is used to keep per-class proportions close between train and validation.
- **Split check & modeling hypothesis:** The proposed split (`summary.json`: Multi-Label Iterative Group Stratified Split, using iterative-stratification) keeps per-class image prevalence close but not identical: the validation share of images per class ranges from 19.3% (bus) to 21.3% (pottedplant), versus 19.9% overall; at object level it ranges from 18.3% (bus) to 27.2% (pottedplant). Stratification does not remove the imbalance itself (person/diningtable usable objects is 21.7× in train and 22.4× in validation). Hypothesis (untested, no model trained): rare classes such as diningtable, sofa and bus will get lower and noisier AP, so class-aware sampling or loss weighting may be needed.
### 2. High Variance in Bounding Box Scales
- **Evidence:** `tables/bbox_summary.csv`, `figures/bbox_size_distribution.png`.
- **Observation:** The relative bounding-box area ranges drastically, with a median of 6.63% (0.066336). Crucially, the bottom 10% (p10) of objects occupy only 0.56% (0.005568) of the total image area. These statistics cover all 15,662 boxes including difficult ones; for valid non-difficult boxes only, the median is 9.64% and p10 is 1.01%.
- **Modeling Hypothesis:** Naive fixed-size downsampling (e.g., resizing to 320x320 for SSDLite320) risks degrading these small objects below the feature extractor's resolution threshold. After resizing to 320×320 for SSDLite320-MobileNetV3, 1,519 boxes (9.7%) would have a shorter side below 16 px, and 17.2% of boxes cover less than 1% of the image. Hypothesis (to be tested after training): small-object AP will be the weakest; this should be checked by reporting AP by box size. A higher-resolution, FPN-based baseline (e.g. torchvision `fasterrcnn_resnet50_fpn`) can serve as a comparison point.
### 3. Multi-Object Scenes and Clutter Density
- **Evidence:** `figures/objects_per_image.png`, `tables/split_summary.csv`, `tables/objects.csv`.
- **Observation:** Objects per image (all XML annotations, including difficult) average 3.126 (median 2.0, max 42); counting valid non-difficult objects only, the mean is 2.516 and the max is 30. The distribution is mostly sparse: 37.2% of images have one object, 60.5% have at most two, and only 4.8% have ten or more. The maximum is image `004349` (42 `bird` boxes, 12 of them difficult).
- **Modeling Hypothesis:** The dataset is multi-label: 43.96% of images contain two or more distinct classes (computed from `tables/objects.csv`). This is why the split uses multi-label iterative stratification instead of single-label stratification (one holdout split, not cross-validation). The long tail of crowded images means the NMS threshold should be tuned at inference.
### 4. Annotation Integrity and Visual Quality Inspection
- **Evidence:** `tables/annotation_issues.csv`, `figures/ground_truth_examples.png`, `example_image_ids.json`, `tables/exact_duplicate_images.csv`.
- **Observation:** Automated checks found 0 annotation issues (`annotation_issues.csv` contains only its header; `file_failures.csv` is empty). The 9 example images in `example_image_ids.json` (`001842`, `000417`, `004494`, `003983`, `003634`, `002300`, `001691`, `008914`, `001441`) were visually checked in `ground_truth_examples.png`, and their boxes look consistent with the objects. This is a spot check of 9 of 5,011 images, not a full audit. Per the VOC2007 guidelines, `truncated` (8,144 objects) marks objects with more than 15–20% of their extent outside the box, and `difficult` (3,054 objects) marks hard-to-recognise objects that VOC evaluation ignores. There are 3 exact-pixel duplicate groups (`000338/007284`, `000949/005042`, `008037/009623`), each kept within a single split.
### 5. Split Integrity, Duplicates, and Leakage Audit
- **Evidence:** `quality_checks.json`, `tables/near_duplicate_candidates.csv`, `figures/near_duplicate_candidates_preview.png`, `figures/split_class_prevalence.png`.
- **Observation:** The proposed split has 4,013 training and 998 validation images, and all 20 classes appear in both (`quality_checks.json`). The smallest validation class is sheep, with 19 images, so its AP will be noisy.
- **Duplicate Review:** The 3 perceptual-hash candidates (dHash distance $\le 4$) were visually reviewed in `figures/near_duplicate_candidates_preview.png`. Decisions are recorded here; the generated CSV is left unchanged.
    - `002765`/`003183` and `001436`/`002765` are different photographs (dHash false positives). Decision: not duplicates, no action needed (all are in train anyway).
    - `000863` (val) / `008364` (train), distance 3, is the same photograph (`000863` has an extra scan border): a true near-duplicate across splits. It affects 1 of 998 validation images, so the leakage is small but real. Decision: put both images in the same split in the next run.

## Review record
- Reviewed by: Nguyễn Hữu Cầu
- Review date: 2026-10-06
- Candidate-pair decisions / unresolved issues: 
  - Reviewed the 3 perceptual-hash candidate pairs: 2 false positives (no action) and 1 true near-duplicate.
  - Unresolved: `000863`/`008364` is split across train and val; it must be grouped into one split in the next run.
- Student edits and verification of assisted notebook/text: 
  - Applied Multi-Label Iterative Group Stratified Split (iterative-stratification library) in the notebook to build the proposed split (`summary.json` → `split_method`).
  - Checked the output tables and figures against each other: per-class validation image share is 19.3–21.3% (close, not perfect). No model was trained, so there are no evaluation metrics.

The split remains proposed until review. No model was trained. No instructor approval is claimed.

## Sources
- dataset: https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/
- published_statistics: https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/dbstats.html
- annotation_and_evaluation: https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/htmldoc/index.html
- archive_and_checksum: https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/voc.html
