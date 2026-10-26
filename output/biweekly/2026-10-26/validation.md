# Sanity check, update of 2026-10-26

Prediction holdouts (not used for tuning; Worlds is not included).

| holdout | series | model_logloss | baseline_logloss | diff (model-base) | diff_se | model_acc | baseline_acc |
|---|---|---|---|---|---|---|---|
| MSI | 20 | 0.54 | 0.622 | -0.083 | 0.054 | 0.75 | 0.7 |
| EWC | 28 | 0.639 | 0.724 | -0.084 | 0.054 | 0.536 | 0.464 |
| LCK playoffs | 10 | 0.59 | 0.593 | -0.003 | 0.01 | 0.8 | 0.8 |
| LPL playoffs | 14 | 0.686 | 0.688 | -0.003 | 0.004 | 0.5 | 0.5 |
| LEC playoffs | 8 | 0.819 | 0.832 | -0.013 | 0.02 | 0.375 | 0.375 |
| LCS playoffs | 10 | 0.626 | 0.628 | -0.002 | 0.011 | 0.5 | 0.5 |
| LCP playoffs | 6 | 0.558 | 0.594 | -0.036 | 0.021 | 0.833 | 0.833 |
| CBLOL playoffs | 10 | 0.592 | 0.61 | -0.018 | 0.011 | 0.8 | 0.8 |
| POOLED international | 48 | 0.598 | 0.681 | -0.084 | 0.038 | 0.625 | 0.562 |
| POOLED domestic | 58 | 0.648 | 0.658 | -0.01 | 0.005 | 0.621 | 0.621 |
| POOLED all | 106 | 0.625 | 0.669 | -0.043 | 0.018 | 0.623 | 0.594 |
