# Final Error Analysis

Best model by validation macro F1: `svm`.

| Model | Valid Macro F1 | Test Macro F1 |
| --- | ---: | ---: |
| logistic_regression | 0.5981 | 0.6292 |
| svm | 0.6918 | 0.6264 |
| lightweight_cnn | 0.4339 | 0.6668 |
| phowhisper_base | 0.6623 | 0.7113 |

## Test Error Summary

- Test predictions analyzed: 45.
- Correct predictions: 28.
- Errors: 17.

## Confusion Patterns

- `Southern->Northern`: 8
- `Northern->Southern`: 6
- `Central->Southern`: 2
- `Northern->Central`: 1

Sample-level errors are saved to `outputs/metrics/final_sample_errors.csv`.

Notes: SVM confidence is a decision margin, not a calibrated probability. PhoWhisper confidence is softmax probability.
