| setting | best epoch | threshold | val precision | val recall | val F1 |
|---|---:|---:|---:|---:|---:|
| lexicon only (no model) | - | - | 0.9915 | 0.94 | 0.9651 |
| XLM-R, raw text | 4 | 0.95 | 0.9659 | 0.919 | 0.9419 |
| XLM-R, normalized text | 4 | 0.93 | 0.9608 | 0.953 | 0.9569 |
| XLM-R, normalized + augmentation (full) | 3 | 0.92 | 0.9557 | 0.9449 | 0.9503 |

Chosen on validation: **XLM-R, normalized text**
