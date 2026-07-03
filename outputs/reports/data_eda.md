# Minimal Data EDA

Phase 3 is intentionally limited to validation needed before the traditional MFCC baseline.

## Split And Class Counts

| Split | Northern | Central | Southern |
| --- | ---: | ---: | ---: |
| train | 3708 | 3416 | 3854 |
| valid | 486 | 487 | 485 |
| test | 486 | 487 | 485 |

## Duration Summary

| Source | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: |
| Original selected audio | 1.05s | 19.12s | 19.16s | 32.24s |
| Preprocessed audio | 16.00s | 16.00s | 16.00s | 16.00s |

## Validation Summary

- Preprocessed files: 13894.
- Files with exact 16 kHz / 16 s shape: 13894.
- Logged preprocessing issues: 0.
- Speaker split validation remains inherited from Phase 1.
- Full figures are deferred; this project is moving to the Phase 4 MFCC baseline after this minimal check.
