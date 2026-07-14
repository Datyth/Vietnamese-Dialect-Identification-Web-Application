# Phase 11 PhoWhisper + CNN Residual Fusion Report

This experiment keeps the PhoWhisper encoder frozen and uses the trained E2 EfficientNetB0-style log-Mel branch as a residual acoustic correction. The residual-gated fusion is `z = g + beta * r(g,l) * P(l)`, where the PhoWhisper baseline head is warm-started from the frozen PhoWhisper checkpoint.

| Split | Accuracy | Macro F1 | Central Recall | Central F1 | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 0.9093 | 0.9069 | 0.8232 | 0.8632 | 0.2456 |
| valid | 0.8422 | 0.8395 | 0.7125 | 0.7669 | 0.4241 |
| test | 0.8128 | 0.8085 | 0.6489 | 0.7223 | 0.4920 |

Best epoch by `hybrid_macro_central`: 5.
Best validation score: 0.8177.
Training device: `mps`.
Fusion type: `residual_gated`.
Beta init: 0.1000.
Beta learned: 0.0928.
Test gate mean: 0.4189145188926833.
PhoWhisper encoder trainable parameters: 0.
EfficientNet local encoder trainable parameters: 69400.
EfficientNet trainable child modules: `5, 6`.
PhoWhisper head warm-start: `True`.
EfficientNet checkpoint: `outputs/models/e2_efficientnetb0_logmel.pt`.
Checkpoint: `outputs/models/e8_whisper_cnn_residual_fusion.pt`.

## Central Dialect Focus

- Test Central recall: 0.6489.
- Test Central F1: 0.7223.
- Central -> Northern errors: 68.
- Central -> Southern errors: 103.

The model predicts only the three dataset-defined regional dialect labels. It does not infer hometown, identity, ethnicity, or personal background.
