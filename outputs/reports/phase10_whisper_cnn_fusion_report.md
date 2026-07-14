# Phase 10 PhoWhisper + CNN Fusion Report

This experiment combines a frozen PhoWhisper encoder branch with a lightly fine-tuned trained E2 EfficientNetB0-style log-Mel branch. Only the selected CNN tail blocks plus the projection, fusion, and classification head are trained.
The decoder is not used, and no ASR transcript is generated.

| Split | Accuracy | Macro F1 | Central Recall | Central F1 | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 0.9061 | 0.9034 | 0.8065 | 0.8604 | 0.2514 |
| valid | 0.8244 | 0.8194 | 0.6530 | 0.7277 | 0.4490 |
| test | 0.8025 | 0.7947 | 0.5914 | 0.6849 | 0.5145 |

Best epoch by validation macro F1: 7.
Training device: `mps`.
Fusion type: `gated`.
PhoWhisper encoder trainable parameters: 0.
EfficientNet local encoder trainable parameters: 69400.
EfficientNet trainable child modules: `5, 6`.
EfficientNet checkpoint: `outputs/models/e2_efficientnetb0_logmel.pt`.
Checkpoint: `outputs/models/e7_whisper_cnn_fusion.pt`.

## Central Dialect Focus

- Test Central recall: 0.5914.
- Test Central F1: 0.6849.
- Central -> Northern errors: 85.
- Central -> Southern errors: 114.

The model predicts only the three dataset-defined regional dialect labels. It does not infer hometown, identity, ethnicity, or personal background.
