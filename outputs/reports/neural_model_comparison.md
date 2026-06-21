# Neural Model Comparison

The pretrained PhoWhisper baseline keeps the encoder unchanged and trains only its projector/classifier head. It is not zero-shot: the original ASR model has no Northern/Central/Southern output head.

| Model | Valid Accuracy | Valid Macro F1 | Test Accuracy | Test Macro F1 | Model Size (MB) | Latency (s/sample) | Device |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Custom CNN | 0.4222 | 0.4339 | 0.6667 | 0.6668 | 0.10 | N/A | cpu |
| PhoWhisper (pretrained, frozen encoder) | 0.6889 | 0.6720 | 0.8000 | 0.7972 | 79.08 | 0.0639 | mps |
| PhoWhisper (fine-tuned) | 0.6667 | 0.6623 | 0.7111 | 0.7113 | 79.08 | 0.0680 | mps |

All rows use the same train/validation/test metadata splits. Model selection remains based on validation macro F1; test metrics are not used to select a model.
