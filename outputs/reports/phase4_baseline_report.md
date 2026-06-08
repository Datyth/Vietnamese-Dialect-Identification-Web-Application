# Phase 4 MFCC Baseline Report

Traditional baselines use MFCC mean/std features from Phase 2 fixed-length audio.

| Model | Split | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| logistic_regression | valid | 0.6000 | 0.5981 |
| logistic_regression | test | 0.6222 | 0.6292 |
| svm | valid | 0.6889 | 0.6918 |
| svm | test | 0.6222 | 0.6264 |

Best model by validation macro F1: svm.

Confusion matrices are saved as CSV files under `outputs/metrics/`.
