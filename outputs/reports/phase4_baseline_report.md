# Phase 4 MFCC Baseline Report

Traditional baselines use MFCC mean/std features from Phase 2 fixed-length audio.

| Model | Split | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| logistic_regression | valid | 0.5617 | 0.5612 |
| logistic_regression | test | 0.5466 | 0.5463 |
| svm | valid | 0.5686 | 0.5688 |
| svm | test | 0.5590 | 0.5592 |

Best model by validation macro F1: svm.

Confusion matrices are saved as CSV files under `outputs/metrics/`.
