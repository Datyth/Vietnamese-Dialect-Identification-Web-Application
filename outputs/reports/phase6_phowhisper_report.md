# Phase 6 PhoWhisper-base Report

PhoWhisper-base was fine-tuned end-to-end for 3-class dialect classification using preprocessed 16 kHz / 16 s audio.

| Split | Accuracy | Macro F1 | Loss |
| --- | ---: | ---: | ---: |
| train | 0.9933 | 0.9933 | 0.0251 |
| valid | 0.6667 | 0.6623 | 1.0027 |
| test | 0.7111 | 0.7113 | 0.7731 |

Best epoch by validation macro F1: 3.
Training device: `mps`.
Checkpoint: `outputs/models/phowhisper_dialect.pt`.

## Model Size

- Model ID: `vinai/PhoWhisper-base`.
- Published parameter count estimate: 74,000,000.
- Hugging Face repository size estimate: 294 MB.
- PyTorch weights size estimate: 290 MB.
- Local checkpoint size: 79.08 MB.

## Latency Estimate

- Samples measured: 5.
- Mean seconds per sample: 0.0680.

Confusion matrices and test predictions are saved under `outputs/metrics/`.
