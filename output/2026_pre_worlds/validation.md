# Validation — snapshot 2026_pre_worlds

## Final validation numbers (seed range 1450-1550, international K x1.0, floor 0.4)

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

All experiments train only on data before each cutoff and predict with frozen ratings. Groups: `international` = MSI + EWC; `domestic` = the six season-final playoffs, each trained on everything before that playoff started; `all` = both together. Worlds is not used. `*_se` is the bootstrap SE of that pooled log-loss; `diff_se` is the bootstrap SE of the paired gap to the plain-Elo baseline over all holdouts. Experiments change one setting at a time from the defaults above.

## 1. Seed range grid

| variant | international_logloss | international_se | domestic_logloss | domestic_se | all_logloss | all_se | all_acc | baseline_all | diff_all (model-base) | diff_se |
|---|---|---|---|---|---|---|---|---|---|---|
| seed 1500-1500 | 0.594 | 0.054 | 0.65 | 0.04 | 0.625 | 0.032 | 0.632 | 0.669 | -0.044 | 0.016 |
| seed 1450-1550 | 0.598 | 0.059 | 0.648 | 0.042 | 0.625 | 0.034 | 0.623 | 0.669 | -0.043 | 0.018 |
| seed 1400-1600 | 0.603 | 0.063 | 0.646 | 0.044 | 0.626 | 0.037 | 0.613 | 0.669 | -0.042 | 0.019 |
| seed 1350-1650 | 0.61 | 0.068 | 0.645 | 0.046 | 0.629 | 0.039 | 0.613 | 0.669 | -0.04 | 0.021 |
| seed 1300-1700 | 0.618 | 0.073 | 0.644 | 0.049 | 0.633 | 0.042 | 0.604 | 0.669 | -0.036 | 0.024 |

## 2. Double counting: domestic multipliers flattened (league offset still on)

| variant | international_logloss | international_se | domestic_logloss | domestic_se | all_logloss | all_se | all_acc | baseline_all | diff_all (model-base) | diff_se |
|---|---|---|---|---|---|---|---|---|---|---|
| seed 1450-1550, current multipliers | 0.598 | 0.059 | 0.648 | 0.042 | 0.625 | 0.034 | 0.623 | 0.669 | -0.043 | 0.018 |
| seed 1450-1550, all multipliers 1.0 | 0.6 | 0.058 | 0.648 | 0.042 | 0.626 | 0.034 | 0.623 | 0.669 | -0.042 | 0.017 |
| seed 1350-1650, current multipliers | 0.61 | 0.068 | 0.645 | 0.046 | 0.629 | 0.039 | 0.613 | 0.669 | -0.04 | 0.021 |
| seed 1350-1650, all multipliers 1.0 | 0.617 | 0.066 | 0.645 | 0.047 | 0.632 | 0.039 | 0.613 | 0.669 | -0.036 | 0.02 |

## 3. International K multiplier

| variant | international_logloss | international_se | domestic_logloss | domestic_se | all_logloss | all_se | all_acc | baseline_all | diff_all (model-base) | diff_se |
|---|---|---|---|---|---|---|---|---|---|---|
| intl K x1.0 | 0.598 | 0.059 | 0.648 | 0.042 | 0.625 | 0.034 | 0.623 | 0.669 | -0.043 | 0.018 |
| intl K x1.5 | 0.604 | 0.061 | 0.652 | 0.044 | 0.63 | 0.036 | 0.604 | 0.672 | -0.042 | 0.017 |
| intl K x2.0 | 0.611 | 0.065 | 0.657 | 0.046 | 0.636 | 0.038 | 0.604 | 0.676 | -0.04 | 0.017 |

## 4. Age-weight floor

| variant | international_logloss | international_se | domestic_logloss | domestic_se | all_logloss | all_se | all_acc | baseline_all | diff_all (model-base) | diff_se |
|---|---|---|---|---|---|---|---|---|---|---|
| floor 0.25 | 0.596 | 0.058 | 0.649 | 0.041 | 0.625 | 0.034 | 0.623 | 0.668 | -0.043 | 0.018 |
| floor 0.4 | 0.598 | 0.059 | 0.648 | 0.042 | 0.625 | 0.034 | 0.623 | 0.669 | -0.043 | 0.018 |
| floor 0.6 | 0.601 | 0.06 | 0.646 | 0.043 | 0.625 | 0.035 | 0.623 | 0.669 | -0.044 | 0.018 |

## 5. Full-season top 20 side by side

| rank | team (1450-1550) | pts (1450-1550) | team (400-2000) | pts (400-2000) |
|---|---|---|---|---|
| 1 | Gen.G | 1794 | Gen.G | 1936 |
| 2 | Bilibili Gaming | 1747 | Hanwha Life Esports | 1890 |
| 3 | Hanwha Life Esports | 1714 | Dplus Kia | 1800 |
| 4 | Dplus Kia | 1704 | T1 | 1773 |
| 5 | T1 | 1636 | Bilibili Gaming | 1765 |
| 6 | Anyone's Legend | 1634 | G2 Esports | 1535 |
| 7 | G2 Esports | 1601 | Karmine Corp | 1473 |
| 8 | Ninjas in Pyjamas.CN | 1579 | Anyone's Legend | 1373 |
| 9 | Karmine Corp | 1576 | KT Rolster | 1304 |
| 10 | KT Rolster | 1576 | LYON (2024 American Team) | 1300 |
| 11 | Team Liquid | 1571 | Team Liquid | 1203 |
| 12 | Invictus Gaming | 1530 | JD Gaming | 1192 |
| 13 | LYON (2024 American Team) | 1528 | Top Esports | 1110 |
| 14 | Top Esports | 1519 | Ninjas in Pyjamas.CN | 1044 |
| 15 | JD Gaming | 1514 | Invictus Gaming | 1042 |
| 16 | LGD Gaming | 1506 | Team WE | 1004 |
| 17 | BNK FEARX | 1485 | BNK FEARX | 976 |
| 18 | Team Secret Whales | 1480 | LGD Gaming | 968 |
| 19 | Cloud9 | 1475 | Team Secret Whales | 968 |
| 20 | Movistar KOI | 1465 | Cloud9 | 949 |

## 5b. Biggest rank changes between the two seed ranges

| team | rank (1450-1550) | rank (400-2000) | change |
|---|---|---|---|
| Ultra Prime | 27 | 47 | -20 |
| Los Ratones | 33 | 51 | -18 |
| Karmine Corp Blue | 39 | 57 | -18 |
| Oh My God | 31 | 48 | -17 |
| Sentinels | 43 | 32 | 11 |
| Team WE | 23 | 16 | 7 |
| Deep Cross Gaming | 48 | 41 | 7 |
| FlyQuest | 40 | 34 | 6 |
| Ninjas in Pyjamas.CN | 8 | 14 | -6 |
| GAM Esports | 45 | 39 | 6 |

Spearman rank correlation over all 60 teams: 0.946. 19 of the top 20 teams are the same under both ranges.

## Ablation (helps > 0 means the piece improves all-holdout log-loss)

| variant | international_logloss | domestic_logloss | all_logloss | all_acc | helps |
|---|---|---|---|---|---|
| full model | 0.598 | 0.648 | 0.625 | 0.623 | 0.0 |
| no league offset | 0.677 | 0.655 | 0.665 | 0.566 | 0.04 |
| no Stage 1 team term | 0.599 | 0.648 | 0.626 | 0.632 | 0.001 |
| no Stakes factor | 0.597 | 0.648 | 0.625 | 0.632 | -0.0 |
| no Quality Wins factor | 0.599 | 0.648 | 0.626 | 0.623 | 0.0 |
| no Network factor | 0.599 | 0.648 | 0.626 | 0.623 | 0.0 |
| no International factor | 0.596 | 0.648 | 0.624 | 0.623 | -0.001 |
| network not age-weighted | 0.598 | 0.648 | 0.625 | 0.623 | 0.0 |
| no international K mult | 0.598 | 0.648 | 0.625 | 0.623 | 0.0 |
| no format K mult | 0.59 | 0.65 | 0.623 | 0.642 | -0.002 |
| no age weight | 0.607 | 0.643 | 0.627 | 0.613 | 0.002 |
