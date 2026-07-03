# Whisper-base original encoder + classifier Frozen Encoder Report

Whisper-base original encoder + classifier was evaluated as a pretrained frozen-encoder baseline. Only the projector and 3-class classifier head were trained.

| Split | Accuracy | Macro F1 | Loss |
| --- | ---: | ---: | ---: |
| train | 0.8470 | 0.8438 | 0.4106 |
| valid | 0.8244 | 0.8229 | 0.4441 |
| test | 0.8189 | 0.8163 | 0.4773 |

Best epoch by validation macro F1: 19.
Training device: `mps`.
Training mode: `frozen_encoder`.
Trainable parameters: 132,099 of 20,722,691.
Checkpoint: `outputs/models/e6_whisper_base_frozen_encoder.pt`.

## Model Size

- Model ID: `openai/whisper-base`.
- Published parameter count estimate for base-size Whisper family: 74,000,000.
- Hugging Face repository size estimate for base-size checkpoint: 294 MB.
- PyTorch weights size estimate for base-size checkpoint: 290 MB.
- Local checkpoint size: 79.08 MB.

## Latency Estimate

- Samples measured: 5.
- Mean seconds per sample: 0.0863.

Confusion matrices and test predictions are saved under `outputs/metrics/`.
