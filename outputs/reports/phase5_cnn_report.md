# Phase 5 Lightweight CNN Report

The CNN uses standardized log-Mel spectrograms from Phase 2 fixed-length audio and is trained from scratch.

| Split | Accuracy | Macro F1 | Loss |
| --- | ---: | ---: | ---: |
| train | 0.7900 | 0.7846 | 0.5366 |
| valid | 0.4222 | 0.4339 | 0.9610 |
| test | 0.6667 | 0.6668 | 0.7248 |

Best epoch by validation macro F1: 13.
Training device: `cpu`.
Checkpoint: `outputs/models/lightweight_cnn_logmel.pt`.

## Baseline Comparison

- Best Phase 4 validation macro F1: 0.6918.
- Phase 5 CNN validation macro F1: 0.4339.

Confusion matrices and the per-epoch training log are saved under `outputs/metrics/`.
