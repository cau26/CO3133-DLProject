# Preliminary EDA notes — review before submission

Run ID: 20261006T174647_307482Z
Prepared for review by Nguyễn Hữu Cầu and Trần Gia Lâm.

## Facts computed by this run
- Successfully decoded images: 5011.
- XML object entries (including difficult): 15662.
- Difficult object entries: 3054.
- Valid non-difficult objects: 12608.
- Proposed train / validation images: 4009 / 1002.
- Valid non-difficult training objects: 10191.
- Most frequent class by valid non-difficult objects: person (4690).
- Least frequent class by the same definition: diningtable (215).
- Objects per image: mean 3.126, median 2.0, maximum 42.
- Median relative area of valid boxes: 0.066336.
- Annotation issues: 0; perceptual-hash candidates: 3.

## Interpretation to be written by Cầu
### 1. Severe Class Imbalance across Categories
- **Evidence:** `tables/class_counts.csv`, `figures/class_distribution.png`.
- **Observation:** Across the 5,011 trainval images, there are 20 object classes with significant instance imbalance. The dominant category is `person` with 4,690 usable non-difficult instances (appearing in 2,008 images, representing ~40.1% of all images). In contrast, the least frequent category is `diningtable` with only 215 usable instances (200 images), followed by `bus` (229 instances) and `sofa` (248 instances). The ratio between the most frequent and least frequent class is approximately 21.8:1. In addition, certain classes contain a very high fraction of difficult objects (e.g., `chair` has 634 difficult objects out of 1,432 total annotations, ~44.3%; `sofa` has 177 difficult out of 425, ~41.6%).
- **Modeling Hypothesis:** Models evaluated with standard mean Average Precision (mAP) may achieve skewed performance where high precision on prevalent classes (`person`, `car`) masks poor generalization on tail classes (`diningtable`, `sheep`, `cow`). Training may suffer from gradient dominance by frequent classes; focal loss or class-aware sampling strategies should be investigated during model development.
### 2. High Variance in Bounding Box Scales and Aspect Ratios
- **Evidence:** `tables/bbox_summary.csv`, `figures/bbox_size_distribution.png`.
- **Observation:** Object bounding boxes exhibit substantial variation in both physical pixel size and relative image area. The relative bounding-box area ranges from a minimum of 0.032% ($0.00032$) to a maximum of 100.0% ($1.0$), with a median of 6.63% ($0.066336$). The bottom 10% of objects occupy less than 0.56% ($0.005568$) of the image area (bounding dimensions down to $26 \times 33$ pixels or even $5 \times 5$ pixels). Conversely, the top 10% exceed 51.0% ($0.510092$) of the total image area. Box aspect ratios ($\text{width}/\text{height}$) range from 0.078 to 15.03 (median 0.849), demonstrating wide structural diversity from tall vertical objects (`person`) to elongated horizontal ones (`train`, `boat`).
- **Modeling Hypothesis:** Downsampling or naive fixed-size resizing (e.g., $300 \times 300$ for SSDLite or $800 \times 800$ for Faster R-CNN) risks degrading small objects below feature extractor resolution thresholds, especially without Feature Pyramid Networks (FPN). Anchors or default boxes must be configured to cover small scales ($\le 32 \times 32$ pixels) as well as extreme aspect ratios ($1:3$ to $3:1$).
### 3. Multi-Object Scenes and Clutter Density
- **Evidence:** `figures/objects_per_image.png`, `tables/split_summary.csv`.
- **Observation:** The number of annotated objects per image averages 3.126 (median 2.0), but exhibits a heavy right tail extending up to 42 objects in a single image. While single-object images exist, a substantial proportion contains dense clusters of objects (e.g., street scenes with multiple pedestrians, cyclists, and cars, or indoor scenes with multiple chairs around a table).
- **Modeling Hypothesis:** Crowded scenes require robust Non-Maximum Suppression (NMS) thresholds and sufficient proposal capacity (e.g., top-k candidate boxes before NMS) to prevent adjacent objects of the same class from being suppressed as duplicates.
### 4. Image Dimensions and Aspect Ratio Diversity
- **Evidence:** `figures/image_dimensions.png`, `tables/images.csv`.
- **Observation:** The dataset contains non-uniform image resolutions, predominantly clustered around $500 \times 375$ (landscape) and $375 \times 500$ (portrait), with aspect ratios varying from ~0.67 to ~1.50. None of the raw images are natively square.
- **Modeling Hypothesis:** Preprocessing pipelines must incorporate aspect-ratio preserving padding or multi-scale jittering rather than anisotropic stretching, which could distort geometric aspect ratios and degrade detector localization accuracy.
### 5. Annotation Integrity and Visual Quality Inspection
- **Evidence:** `tables/annotation_issues.csv`, `figures/ground_truth_examples.png`, `example_image_ids.json`.
- **Observation:** Automated checks confirmed 0 coordinate anomalies or syntax errors (`annotation_issues.csv` is completely empty; all boxes satisfy $1 \le x_{\min} \le x_{\max} \le W$ and $1 \le y_{\min} \le y_{\max} \le H$). Nine sample images were visually inspected (`001842`, `000417`, `004494`, `003983`, `003634`, `002300`, `001691`, `008914`, `001441`). Bounding boxes tightly enclose target objects. Truncated objects (e.g., vehicles partially outside frame) are appropriately labeled with `truncated=1`, and occluded/heavily blurred objects are consistently assigned the `difficult=1` flag.
- **Modeling Hypothesis:** The official VOC metric ignores `difficult=1` instances during evaluation. For training, retaining or excluding difficult objects is an experimental variable; treating heavily occluded difficult instances as background vs. positive training targets could impact false positive rates.
### 6. Split Integrity, Duplicates, and Leakage Audit
- **Evidence:** `quality_checks.json`, `tables/exact_duplicate_images.csv`, `tables/near_duplicate_candidates.csv`.
- **Observation:** The proposed 80/20 grouped random split yields 4,009 training images and 1,002 validation images, perfectly meeting the target partition. All 20 classes are present in both splits (`all_20_classes_in_train = True`, `all_20_classes_in_validation = True`). 
  - *Exact duplicates:* 3 duplicate groups (6 images total: `000338`/`007284`, `000949`/`005042`, `008037`/`009623`) were identified via exact SHA-256 pixel hashes. The grouping algorithm successfully placed each pair entirely inside a single split (the first two pairs in `train`, the third in `val`), preventing zero cross-split leakage.
  - *Near duplicates:* The dHash check ($\le 4$) identified 3 candidate pairs (`000863`/`008364`, `002765`/`003183`, `001436`/`002765`). Critically, all 3 candidate pairs belong strictly to the `train` partition (`cross_split = False`, 0 cross-split candidates).
- **Modeling Hypothesis:** The proposed split is clean of detected data leakage between training and validation sets. However, before final benchmarking, potential overlap between the `trainval` pool and the official `test` set must be audited once the test set is released.

## Review record
- Reviewed by: Nguyễn Hữu Cầu
- Review date: 2026-10-06
- Candidate-pair decisions / unresolved issues: 
  - Reviewed the 3 perceptual-hash candidate pairs from `near_duplicate_candidates.csv`. All 3 pairs are within the training split (`cross_split = False`), confirming no train-val data leakage.
  - No corrupted images or out-of-bound annotations were encountered.
- Student edits and verification of assisted notebook/text: 
  - Executed `A2_EDA_VOC2007.ipynb` on Google Colab CPU runtime.
  - Verified MD5 checksum (`c52e279531787c972589f7e41ab4ae64`) and confirmed 5,011 decoded images.
  - Inspected generated figures, verified statistical distribution CSVs, and drafted technical observations based strictly on run outputs.

The split remains proposed until review. No model was trained. No instructor approval is claimed.

## Sources
- dataset: https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/
- published_statistics: https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/dbstats.html
- annotation_and_evaluation: https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/htmldoc/index.html
- archive_and_checksum: https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/voc.html
