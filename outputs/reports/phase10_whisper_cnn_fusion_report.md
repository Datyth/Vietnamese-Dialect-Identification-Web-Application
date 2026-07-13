# Phase 10 PhoWhisper + CNN Fusion Report

This experiment combines a frozen PhoWhisper encoder branch with a frozen trained E2 EfficientNetB0-style log-Mel branch. Only the projection, fusion, and classification head are trained. The decoder is not used, and no ASR transcript is generated.

| Split | Accuracy | Macro F1 | Central Recall | Central F1 | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 0.8991 | 0.8960 | 0.7939 | 0.8479 | 0.2925 |
| valid | 0.8121 | 0.8072 | 0.6427 | 0.7154 | 0.4858 |
| test | 0.7874 | 0.7802 | 0.5811 | 0.6659 | 0.5291 |

Best epoch by validation macro F1: 5.
Training device: `mps`.
Fusion type: `concat`.
PhoWhisper encoder trainable parameters: 0.
EfficientNet local encoder trainable parameters: 0.
EfficientNet checkpoint: `outputs/models/e2_efficientnetb0_logmel.pt`.
Checkpoint: `outputs/models/e7_whisper_cnn_fusion.pt`.

## Central Dialect Focus

- Test Central recall: 0.5811.
- Test Central F1: 0.6659.
- Central -> Northern errors: 85.
- Central -> Southern errors: 119.

The model predicts only the three dataset-defined regional dialect labels. It does not infer hometown, identity, ethnicity, or personal background.
