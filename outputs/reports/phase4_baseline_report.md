# Phase 4 MFCC Baseline Report

Traditional baselines use MFCC mean/std features from Phase 2 fixed-length audio.

| Model | Split | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| logistic_regression | valid | 0.6667 | 0.6688 |
| logistic_regression | test | 0.6222 | 0.6284 |
| svm | valid | 0.7111 | 0.7115 |
| svm | test | 0.6667 | 0.6773 |

Best model by validation macro F1: svm.

Confusion matrices are saved as CSV files under `outputs/metrics/`.
