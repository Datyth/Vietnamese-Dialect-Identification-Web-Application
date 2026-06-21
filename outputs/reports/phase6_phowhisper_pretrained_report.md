# Phase 6 PhoWhisper-base Pretrained Encoder Report

PhoWhisper-base was evaluated as a pretrained frozen-encoder baseline. Only the projector and 3-class classifier head were trained.

| Split | Accuracy | Macro F1 | Loss |
| --- | ---: | ---: | ---: |
| train | 0.9433 | 0.9428 | 0.1307 |
| valid | 0.6889 | 0.6720 | 1.0789 |
| test | 0.8000 | 0.7972 | 0.8316 |

Best epoch by validation macro F1: 8.
Training device: `mps`.
Training mode: `frozen_encoder`.
Trainable parameters: 132,099 of 20,722,691.
Checkpoint: `outputs/models/phowhisper_pretrained_frozen_encoder.pt`.

## Model Size

- Model ID: `vinai/PhoWhisper-base`.
- Published parameter count estimate: 74,000,000.
- Hugging Face repository size estimate: 294 MB.
- PyTorch weights size estimate: 290 MB.
- Local checkpoint size: 79.08 MB.

## Latency Estimate

- Samples measured: 5.
- Mean seconds per sample: 0.0639.

Confusion matrices and test predictions are saved under `outputs/metrics/`.
