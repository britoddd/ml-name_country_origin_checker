# EDA report — name-to-nationality dataset
## Raw aggregated tables
- forenames.csv: **12,443,052** rows across 104 countries
- surnames.csv: **21,079,771** rows across 104 countries
- distinct forenames/country: min 223 (KH) … max 760,327 (SA)
- distinct surnames/country: min 216 (KH) … max 843,532 (EG)

## Generated splits
- classes: **104**  |  train/val/test = 3,624,986 / 776,782 / 776,784
- train imbalance (max/min): **1.751x** (max AE=35,000, min KH=19,986)
- name length: mean 13.22 chars, p95 19 (cap is max_len=30)
- tokens per name: {'2': 274417, '3': 23798, '4': 1666, '5': 113, '6': 6}

## Multi-country ambiguity
- names spanning >1 country: **174,564** (mean breadth 2.711, max 30)
- broadest example: `ali ali` → 30 countries
