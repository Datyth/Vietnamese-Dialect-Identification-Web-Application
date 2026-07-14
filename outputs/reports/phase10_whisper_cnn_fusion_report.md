# Phase 10 PhoWhisper + CNN Fusion Report

This experiment combines a frozen PhoWhisper encoder branch with a frozen trained E2 EfficientNetB0-style log-Mel branch. Only the projection, fusion, and classification head are trained. The decoder is not used, and no ASR transcript is generated.

| Split | Accuracy | Macro F1 | Central Recall | Central F1 | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 0.9078 | 0.9048 | 0.7974 | 0.8590 | 0.2523 |
| valid | 0.8237 | 0.8192 | 0.6571 | 0.7356 | 0.4724 |
| test | 0.7936 | 0.7866 | 0.5893 | 0.6809 | 0.5483 |

Best epoch by validation macro F1: 11.
Training device: `mps`.
Fusion type: `gated`.
PhoWhisper encoder trainable parameters: 0.
EfficientNet local encoder trainable parameters: 0.
EfficientNet checkpoint: `outputs/models/e2_efficientnetb0_logmel.pt`.
Checkpoint: `outputs/models/e7_whisper_cnn_fusion.pt`.

## Central Dialect Focus

- Test Central recall: 0.5893.
- Test Central F1: 0.6809.
- Central -> Northern errors: 82.
- Central -> Southern errors: 118.

The model predicts only the three dataset-defined regional dialect labels. It does not infer hometown, identity, ethnicity, or personal background.
