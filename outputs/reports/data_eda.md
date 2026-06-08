# Minimal Data EDA

Phase 3 is intentionally limited to validation needed before the traditional MFCC baseline.

## Split And Class Counts

| Split | Northern | Central | Southern |
| --- | ---: | ---: | ---: |
| train | 100 | 100 | 100 |
| valid | 15 | 15 | 15 |
| test | 15 | 15 | 15 |

## Duration Summary

| Source | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: |
| Original selected audio | 2.51s | 15.94s | 16.01s | 31.82s |
| Preprocessed audio | 16.00s | 16.00s | 16.00s | 16.00s |

## Validation Summary

- Preprocessed files: 390.
- Files with exact 16 kHz / 16 s shape: 390.
- Logged preprocessing issues: 0.
- Speaker split validation remains inherited from Phase 1.
- Full figures are deferred; this project is moving to the Phase 4 MFCC baseline after this minimal check.
