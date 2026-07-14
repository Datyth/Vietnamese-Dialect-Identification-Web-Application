# Central Class Data Analysis

## Purpose

This note summarizes the evidence behind why the `Central` dialect class is
harder than `Northern` and `Southern` in the current Vietnamese dialect
identification experiments.

The analysis is technical and dataset-level only. It does not infer a speaker's
hometown, identity, ethnicity, or personal background.

## Source Artifacts

Main sources used:

- `notebooks/e7_central_error_analysis.ipynb`
- `outputs/reports/e7_central_error_analysis.md`
- `outputs/metrics/e7_whisper_cnn_fusion_results.json`
- `outputs/metrics/e7_whisper_cnn_fusion_test_confusion_matrix.csv`
- `outputs/metrics/e8_whisper_cnn_residual_fusion_results.json`
- `outputs/metrics/e8_whisper_cnn_residual_fusion_test_confusion_matrix.csv`
- `outputs/metrics/phowhisper_pretrained_test_confusion_matrix.csv`
- `outputs/metrics/e7_central_error_analysis_audio_features.csv`

The notebook generated figures under:

- `outputs/figures/e7_central_error_analysis/`

## Executive Summary

The current evidence suggests that `Central` is difficult mainly because it is
an overlapping and transitional class in the model's feature space, not because
of a single obvious data-quality issue.

The most important observations are:

- `Central` recall is much lower than its precision in E7.
- Most `Central` errors are pushed toward `Southern`, then `Northern`.
- The class counts and test split are close enough that simple class imbalance
  does not explain the problem.
- Audio quality metadata such as duration, RMS, and audio size look broadly
  similar across classes.
- A balanced sample of simple acoustic features shows substantial overlap.
  In the nearest-centroid check, only `38/80` sampled `Central` examples are
  closest to the `Central` centroid, while `42/80` are closer to a non-Central
  centroid.
- E8 residual-gated fusion improves over E7 on `Central`, but still does not
  beat the frozen PhoWhisper baseline E4.

Practical conclusion: use E4 PhoWhisper as the current best model. Treat E8 as
a useful ablation showing that safer residual fusion helps E7, but the remaining
Central weakness is better addressed through targeted data/error analysis and
possibly light PhoWhisper fine-tuning, not by adding more fusion complexity.

## 1. Model Behavior On Central

### E7 Confusion Matrix

E7 is the gated PhoWhisper + CNN fusion model with frozen PhoWhisper and a
lightly fine-tuned EfficientNetB0-style local branch.

Test confusion matrix:

| True label | Pred Northern | Pred Central | Pred Southern |
| --- | ---: | ---: | ---: |
| Northern | 464 | 16 | 6 |
| Central | 85 | 288 | 114 |
| Southern | 17 | 50 | 418 |

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Northern | 0.8198 | 0.9547 | 0.8821 | 486 |
| Central | 0.8136 | 0.5914 | 0.6849 | 487 |
| Southern | 0.7770 | 0.8619 | 0.8172 | 485 |

Overall:

- Accuracy: `0.8025`
- Macro F1: `0.7947`
- Central -> Northern errors: `85`
- Central -> Southern errors: `114`

The key pattern is high-ish `Central` precision but low `Central` recall.
When E7 predicts `Central`, it is often plausible. The issue is that many true
`Central` samples are not selected as `Central`; they are absorbed by the two
neighboring classes, especially `Southern`.

This is a recall problem more than a precision problem.

### Comparison With E4 And E8

The current strongest baseline is E4 PhoWhisper frozen encoder. E8 is the
residual-gated fusion model added after E7.

| Model | Test Macro F1 | Central Precision | Central Recall | Central F1 | Central -> Northern | Central -> Southern |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E4 PhoWhisper frozen encoder | 0.8348 | 0.8169 | 0.7331 | 0.7727 | 57 | 73 |
| E7 gated fusion | 0.7947 | 0.8136 | 0.5914 | 0.6849 | 85 | 114 |
| E8 residual-gated fusion | 0.8085 | 0.8144 | 0.6489 | 0.7223 | 68 | 103 |

Interpretation:

- E7 hurts `Central` compared with E4. It increases both Central -> Northern
  and Central -> Southern errors.
- E8 improves E7. It reduces Central -> Northern from `85` to `68` and Central
  -> Southern from `114` to `103`.
- E8 still trails E4. Central recall remains about `8.42` percentage points
  lower than E4.

This suggests that the residual design is safer than E7's legacy gated
interpolation, but the local CNN branch still does not provide enough
complementary signal to beat PhoWhisper by itself.

## 2. Dataset Composition

The prepared dataset contains `13,894` supported rows across the three project
labels.

Split distribution:

| Split | Northern | Central | Southern |
| --- | ---: | ---: | ---: |
| Train | 3708 | 3416 | 3854 |
| Validation | 486 | 487 | 485 |
| Test | 486 | 487 | 485 |

Overall metadata summary:

| Label | Samples | Unique speakers | Unique provinces | Male | Female |
| --- | ---: | ---: | ---: | ---: | ---: |
| Northern | 4680 | 3419 | 24 | 3291 | 1389 |
| Central | 4390 | 3022 | 19 | 3176 | 1214 |
| Southern | 4824 | 3101 | 19 | 3480 | 1344 |

Observations:

- The validation and test sets are essentially balanced.
- The training set has fewer `Central` examples than `Northern` and `Southern`,
  but the gap is not large enough to fully explain the large recall gap.
- `Central` has fewer unique speakers than `Northern`, but similar to
  `Southern`.
- All classes have many speakers and low samples per speaker, so the issue does
  not look like a simple speaker dominance problem.

## 3. Province And Speaker Diversity

Province entropy from the notebook:

| Label | Province count | Province entropy | Normalized entropy | Top province | Top count |
| --- | ---: | ---: | ---: | --- | ---: |
| Northern | 24 | 4.3920 | 0.9579 | CaoBang | 357 |
| Central | 19 | 4.1216 | 0.9703 | BinhThuan | 325 |
| Southern | 19 | 4.1969 | 0.9880 | BinhPhuoc | 319 |

Speaker samples:

| Label | Speakers | Mean samples/speaker | Median | Max |
| --- | ---: | ---: | ---: | ---: |
| Northern | 3419 | 1.3688 | 1.0000 | 9 |
| Central | 3022 | 1.4527 | 1.0000 | 9 |
| Southern | 3101 | 1.5556 | 1.0000 | 8 |

Observations:

- `Central` does not have the highest province count.
- `Central` also does not have the highest normalized province entropy.
- The top province is not overwhelmingly dominant in any class.
- Median samples per speaker is `1` for all classes.

So the notebook does not support a simple conclusion like "`Central` is harder
because it is the most province-diverse class." A better explanation is that
some `Central` provinces or sub-regions may sit near the Northern/Southern
decision boundaries, but that requires a more targeted province-level error
analysis.

## 4. Audio Duration And Quality Signals

Metadata-derived audio statistics:

### Original Duration

| Label | Mean | Median | Std | P90 - P10 |
| --- | ---: | ---: | ---: | ---: |
| Northern | 19.1602 | 19.0930 | 6.1210 | 16.3584 |
| Central | 18.8944 | 18.8373 | 6.1371 | 16.6001 |
| Southern | 19.3916 | 19.4396 | 6.6215 | 17.9992 |

### Preprocessed RMS

| Label | Mean | Median | Std | P90 - P10 |
| --- | ---: | ---: | ---: | ---: |
| Northern | 0.0745 | 0.0779 | 0.0093 | 0.0211 |
| Central | 0.0744 | 0.0779 | 0.0092 | 0.0206 |
| Southern | 0.0747 | 0.0779 | 0.0093 | 0.0216 |

### Preprocessed Peak

| Label | Mean | Median | Std | P90 - P10 |
| --- | ---: | ---: | ---: | ---: |
| Northern | 0.6954 | 0.6929 | 0.1935 | 0.5565 |
| Central | 0.6644 | 0.6356 | 0.1735 | 0.4964 |
| Southern | 0.6672 | 0.6562 | 0.1904 | 0.5382 |

### Audio Bytes

| Label | Mean | Median | Std | P90 - P10 |
| --- | ---: | ---: | ---: | ---: |
| Northern | 613168.8410 | 611020.0000 | 195872.7332 | 523467.6000 |
| Central | 604664.3868 | 602837.0000 | 196386.1364 | 531203.6000 |
| Southern | 620575.5626 | 622111.0000 | 211888.7415 | 575972.8000 |

Observations:

- Duration distributions are similar.
- RMS is nearly identical across classes after preprocessing.
- `Central` has a slightly lower peak amplitude on average, but this is not
  large enough by itself to explain the classification gap.
- Audio size is also broadly similar.

Therefore, there is no strong evidence that `Central` is harder simply because
its audio files are shorter, quieter, or lower quality.

## 5. Acoustic Feature Sample

The notebook sampled `80` examples per class and computed simple audio features:

- RMS and peak amplitude.
- Zero-crossing rate.
- Spectral centroid, bandwidth, and flatness.
- Log-Mel mean and standard deviation.
- MFCC summary features.

This is a descriptive diagnostic, not the training representation of E7/E8.
Still, it helps reveal whether the classes are cleanly separable under simple
acoustic cues.

Selected feature means:

| Feature | Northern | Central | Southern |
| --- | ---: | ---: | ---: |
| RMS | 0.0732 | 0.0744 | 0.0767 |
| Peak | 0.6898 | 0.6726 | 0.6536 |
| Zero-crossing rate | 0.0905 | 0.0781 | 0.0867 |
| Spectral centroid mean | 616.4752 | 538.7777 | 593.2555 |
| Spectral bandwidth mean | 518.4663 | 462.9152 | 515.8159 |
| Spectral flatness mean | 0.1140 | 0.0973 | 0.0753 |
| Log-Mel mean | -11.8459 | -11.9704 | -11.1673 |
| Log-Mel std | 4.3333 | 4.5787 | 4.1469 |

Observations:

- `Central` often sits between or near the other classes rather than forming a
  cleanly separated region.
- Some features make `Central` look closer to `Northern`, while others make it
  closer to `Southern`.
- This supports the idea that a single global decision boundary is hard for
  `Central`.

## 6. PCA And Nearest-Centroid Evidence

The PCA plot generated by the notebook shows heavy overlap between all three
classes. The first two components explain:

- PC1: `50.2%`
- PC2: `22.5%`

The key visual pattern is not a clean three-cluster structure. `Central` samples
are mixed with both `Northern` and `Southern`, especially around the dense
middle region of the PCA plot.

Nearest-centroid matrix from the sampled acoustic features:

| True label | Near Northern | Near Central | Near Southern | Own-centroid rate |
| --- | ---: | ---: | ---: | ---: |
| Northern | 15 | 28 | 37 | 0.1875 |
| Central | 10 | 38 | 32 | 0.4750 |
| Southern | 7 | 24 | 49 | 0.6125 |

For `Central`:

- `38/80` are nearest the `Central` centroid.
- `10/80` are nearest the `Northern` centroid.
- `32/80` are nearest the `Southern` centroid.
- `42/80` are closer to a non-Central centroid than to the Central centroid.

This is the strongest notebook-level evidence for the overlap hypothesis.
It also matches the model confusion pattern: Central is more often confused
with Southern than with Northern.

Important nuance: the nearest-centroid diagnostic uses simple acoustic features,
not PhoWhisper embeddings. It should be read as evidence of acoustic overlap,
not as a direct explanation of every model prediction.

## 7. Why E8 Helps But Does Not Solve Central

E8 changes the fusion strategy from legacy gated interpolation to residual-gated
fusion:

```text
z = g + beta * r(g,l) * P(l)
```

Here `g` is the PhoWhisper representation, and the local CNN branch only adds a
learned residual correction. This is safer than E7 because the model starts
near the PhoWhisper baseline and does not force a direct interpolation between
global and local embeddings.

E8 test results:

| True label | Pred Northern | Pred Central | Pred Southern |
| --- | ---: | ---: | ---: |
| Northern | 451 | 19 | 16 |
| Central | 68 | 316 | 103 |
| Southern | 14 | 53 | 418 |

Compared with E7:

- Central -> Northern decreases from `85` to `68`.
- Central -> Southern decreases from `114` to `103`.
- Central recall improves from `0.5914` to `0.6489`.
- Central F1 improves from `0.6849` to `0.7223`.

But E8 still does not reach E4:

- E4 Central recall: `0.7331`
- E8 Central recall: `0.6489`
- E4 Central F1: `0.7727`
- E8 Central F1: `0.7223`

E8's learned beta is close to its initialization:

- beta init: `0.1`
- beta learned: `0.0928`
- test residual gate mean: `0.4189`

This suggests the model uses the CNN branch as a small correction, not as a
dominant source of information. That is a healthy residual behavior, but it
also suggests the CNN branch has limited useful signal for resolving Central.

## 8. Most Likely Explanation

The most likely explanation is a combination of:

1. `Central` is a broad label with internal acoustic variation.
2. Some `Central` samples lie near the Southern/Northern decision boundaries.
3. Simple acoustic features show real overlap, especially Central toward
   Southern.
4. PhoWhisper's pretrained representation handles the overlap better than the
   current CNN fusion variants.
5. E7's gated interpolation may disturb the strong PhoWhisper representation.
   E8 reduces this damage, but does not add enough new information to beat E4.

The data does not strongly support these simpler explanations:

- Severe class imbalance.
- Obvious audio quality problem specific to `Central`.
- Province entropy alone.
- Speaker dominance by a small number of Central speakers.

## 9. Recommended Next Analysis

The next useful analysis should focus on the actual error subsets, not just
aggregate class metrics.

Recommended checks:

1. Build tables for `Central -> Southern` and `Central -> Northern` errors by
   province.
2. Compare error rates per province against province sample counts.
3. Inspect whether the same speakers or source shards appear repeatedly in
   Central errors.
4. Enable the notebook's optional E7 forward section to export prediction
   confidence and fused embeddings.
5. Repeat the embedding/confidence analysis for E8 if needed, especially gate
   and residual magnitude by class.

The most important immediate question is:

> Are Central errors concentrated in specific provinces/sub-regions, or are
> they spread broadly across Central?

If errors are concentrated, targeted data balancing or province-aware sampling
may help. If errors are spread broadly, the issue is probably the class boundary
itself and may require better representation learning.

## 10. Recommended Modeling Actions

Recommended model direction:

1. Keep E4 PhoWhisper frozen encoder as the current best model.
2. Treat E8 as a useful ablation, not as the primary model.
3. Do not add more complex fusion layers until the error subsets are understood.
4. If training another model, prioritize light PhoWhisper fine-tuning over
   additional CNN-fusion complexity.
5. Consider Central-focused loss adjustments only after checking confidence:
   class weighting, focal loss, or a small Central logit bias can improve recall
   but may reduce precision.
6. If using augmentation, target Central boundary cases rather than applying
   generic augmentation equally to all classes.

Potential trade-offs:

- Increasing Central recall may increase false Central predictions for Northern
  and Southern.
- Class weighting may improve Central F1 but reduce overall macro F1 if it
  over-corrects.
- More fusion capacity may overfit and make the model less stable unless the
  local branch provides clearly complementary information.

## 11. Limitations

This analysis has several limitations:

- The acoustic feature sample uses `80` examples per class, not the full
  dataset.
- PCA and nearest-centroid diagnostics use simple handcrafted features, not the
  exact model embeddings.
- The optional E7 forward section was skipped in the notebook, so the current
  report does not include per-sample confidence or fused embedding analysis.
- E8 diagnostics are taken from saved metrics artifacts, not from a new
  notebook-style error analysis.
- The three labels are broad regional classes. Some dialect transitions may be
  inherently ambiguous under this label scheme.

## Final Takeaway

`Central` should be described as the hardest class because its boundary overlaps
with both other classes, especially `Southern`. The current evidence points more
toward acoustic/representation overlap than toward data quantity or audio
quality. E8 confirms that a safer residual fusion can reduce E7's Central
errors, but PhoWhisper alone still remains stronger. The next best step is
province-level and confidence-level analysis of `Central -> Southern` and
`Central -> Northern` errors before changing the architecture again.
