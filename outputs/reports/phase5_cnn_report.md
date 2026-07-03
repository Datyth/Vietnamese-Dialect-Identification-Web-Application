# Phase 5 Lightweight CNN Report

The CNN uses standardized log-Mel spectrograms from Phase 2 fixed-length audio and is trained from scratch.

| Split | Accuracy | Macro F1 | Loss |
| --- | ---: | ---: | ---: |
| train | 0.6218 | 0.6137 | 0.8694 |
| valid | 0.6091 | 0.6052 | 0.8875 |
| test | 0.6187 | 0.6115 | 0.8867 |

Best epoch by validation macro F1: 21.
Training device: `mps`.
Checkpoint: `outputs/models/lightweight_cnn_logmel.pt`.

## Baseline Comparison

- Best Phase 4 validation macro F1: 0.5688.
- Phase 5 CNN validation macro F1: 0.6052.

Confusion matrices and the per-epoch training log are saved under `outputs/metrics/`.
