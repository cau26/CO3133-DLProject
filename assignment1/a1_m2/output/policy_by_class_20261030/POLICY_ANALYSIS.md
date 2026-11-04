# Class-conditional selective prediction

This post-hoc diagnostic replays the preset **C4 seed314** policy on the 2,766-crop prospective cohort. It uses temperature **5.345757234366385** and confidence threshold **0.88**, retaining equality in the accepted set. No fitting, model inference, calibration or policy adaptation takes place. Results refer to one fixed serving model, rather than means over three training seeds.

## Conditioning on the true class

| Class | Accepted / support | Correct / accepted | Review | Coverage | Accepted accuracy |
|---|---:|---:|---:|---:|---:|
| all | 1506/2766 | 1445/1506 | 1260 | 54.45% | 95.95% |
| shirt_blouse | 33/182 | 22/33 | 149 | 18.13% | 66.67% |
| top_tshirt_sweatshirt | 431/856 | 406/431 | 425 | 50.35% | 94.20% |
| jacket | 82/300 | 76/82 | 218 | 27.33% | 92.68% |
| pants | 361/476 | 360/361 | 115 | 75.84% | 99.72% |
| skirt | 106/257 | 100/106 | 151 | 41.25% | 94.34% |
| dress | 493/695 | 481/493 | 202 | 70.94% | 97.57% |

Coverage uses all crops with a given annotated label. Accepted accuracy uses only accepted crops of that true class. These labels are available during offline evaluation, not when routing a new unlabeled input. Shirt/blouse sends 149 of 182 crops to review, but still misclassifies 11 of 33 accepted crops. Pants accepts 361 of 476 crops with one accepted error. The pooled 95.95% accepted accuracy therefore does not describe uniform performance across garment classes.

## Conditioning on the predicted class

| Class | Accepted / support | Correct / accepted | Review | Coverage | Accepted accuracy |
|---|---:|---:|---:|---:|---:|
| all | 1506/2766 | 1445/1506 | 1260 | 54.45% | 95.95% |
| shirt_blouse | 24/183 | 22/24 | 159 | 13.11% | 91.67% |
| top_tshirt_sweatshirt | 428/856 | 406/428 | 428 | 50.00% | 94.86% |
| jacket | 88/284 | 76/88 | 196 | 30.99% | 86.36% |
| pants | 362/484 | 360/362 | 122 | 74.79% | 99.45% |
| skirt | 101/243 | 100/101 | 142 | 41.56% | 99.01% |
| dress | 503/716 | 481/503 | 213 | 70.25% | 95.63% |

Here the groups use the emitted model label. For example, 22 of 24 accepted shirt/blouse outputs are correct (91.67%), while 22 of 33 accepted true shirt/blouse inputs are correct (66.67%). The denominators differ because the latter also includes confidently misclassified shirts assigned to other labels. Accepted accuracy in the predicted-label table is precision among the accepted outputs of each label. Predicted jacket outputs have 76 correct predictions among 88 accepted outputs (86.36%), revealing an operational weakness that the pooled score alone obscures.

The policy refers 1,260 crops for review. Of the 495 errors in the full-cohort model predictions, 434 are referred and 61 escape into the accepted set. Review also includes 826 already correct model predictions. These are routing diagnostics; the study does not measure human correction quality or end-to-end human-assisted accuracy.

## Uncertainty and interpretation

The JSON retains 2,000-replicate source/duplicate-component percentile intervals for each class's coverage and accepted accuracy. They are post-hoc pointwise descriptive intervals, conditional on this fixed model and cohort. They do not provide simultaneous classwise certification, estimate the population of training seeds or transfer the original calibration risk guarantee to the license shift. Small accepted subsets, particularly 24 emitted shirt/blouse predictions, limit precision. Conditioning on the true label also differs from conditioning on a forecast probability; a true-class accepted-accuracy value alone does not directly test a probability-calibration identity.

The threshold remains fixed. Changing it by class using the inspected cohort would be further development, requiring independent calibration and a new untouched assessment population. The current evidence supports reporting the unequal routing/error burden, not claiming the defect has been fixed by this analysis.
