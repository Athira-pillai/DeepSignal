# Phase 7 optional XGBoost benchmark

Run: `20261031T141506.545871Z`  
Run started (UTC): `2026-10-31T14:15:06.545871+00:00`  
Run completed (UTC): `2026-10-31T14:16:30.967370+00:00`  
Provider: `Yahoo Finance`  
Status: **ok**

## Method and scope

- Target: next session adjusted close greater than current session adjusted close.
- Source: Yahoo Finance daily bars, `auto_adjust=True`, 3-year requested period.
- Each successful ticker has expanding train/validation/test folds in its first 80% and a separate final chronological 20% holdout.
- The final holdout is distinct from the walk-forward folds; no settings, weights, or thresholds were tuned on its results.
- Fixed hybrid is retained at 40% LSTM, 40% Random Forest, 10% neutral historical sentiment, and 10% technical evidence.
- XGBoost is included only if enabled in the run configuration.
- Historical point-in-time news is unavailable, so sentiment is neutral for this evaluation.
- The classifier trading figures are simplified long-or-cash close-to-close simulations, not the separate next-open technical backtest.
- Raw data is not archived with this report; the SHA-256 identifies the evaluated OHLCV frame but does not make the provider's historical dataset immutable.

## Configuration

- Features: `return_1d, return_5d, sma20_gap, sma50_gap, rsi, macd, macd_signal, bb_position, volume_change`.
- Random Forest: 250 trees; max depth 7; minimum leaf size 4; balanced class weights; seed 42.
- LSTM: lookback 30 rows; 48 units; up to 6 epochs; early stopping on validation data.
- Costs for simplified trading metrics: 0.06% fee and 0.04% slippage on position changes.
- Decision cutoff is after the current daily close; direction threshold is 0.50 for classifier metrics. The live hybrid's Buy/Sell thresholds are not applied to the evaluation classification table.
- The final holdout model and fold models use fixed RF parameters. No model hyperparameter sweep, threshold search, calibration, adaptive weighting, or No Trade rule was run.

## Data coverage and status

| Ticker | Sector | Status | Raw rows | Labeled rows | Raw data dates | Labeled dates | SHA-256 | Folds | Final holdout |
| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | --- |
| AAPL | Technology | ok | 753 | 703 | 2023-10-31–2026-10-30 | 2024-01-11–2026-10-29 | `c993e7f7d8372896b6572f3d5c83b4cb34455c025121343989bd0b622b621e8c` | 4 | 2026-04-10–2026-10-29 |
| MSFT | Technology | ok | 753 | 703 | 2023-10-31–2026-10-30 | 2024-01-11–2026-10-29 | `b04d3e2b379e4cc92fc257fa39e9ed66584b511d3ff9c6702c02c1a33f6d5e2b` | 4 | 2026-04-10–2026-10-29 |
| JPM | Financials | ok | 753 | 703 | 2023-10-31–2026-10-30 | 2024-01-11–2026-10-29 | `db23ca2155b24539d3efe3b5bf3d7fec928e3a2b1db18a1bae797777b8604856` | 4 | 2026-04-10–2026-10-29 |
| JNJ | Healthcare | ok | 753 | 703 | 2023-10-31–2026-10-30 | 2024-01-11–2026-10-29 | `28bf169d669ad8d738690bef1ae880136af84a8a96a8e9ea25660e421f573c43` | 4 | 2026-04-10–2026-10-29 |
| XOM | Energy | ok | 753 | 703 | 2023-10-31–2026-10-30 | 2024-01-11–2026-10-29 | `fd57fadc81f9ee5e3b76eb69889861ecb5035024f588fe6b3bfaef19e26079ce` | 4 | 2026-04-10–2026-10-29 |

## Fold date boundaries

| Ticker | Fold | Train end | Validation | Test | Train/validation/test rows |
| --- | ---: | --- | --- | --- | ---: |
| AAPL | 1 | 2024-01-11–2025-02-25 | 2025-02-26–2025-05-15 | 2025-05-16–2025-08-07 | 281/56/57 |
| AAPL | 2 | 2024-01-11–2025-05-16 | 2025-05-19–2025-08-07 | 2025-08-08–2025-10-27 | 338/56/56 |
| AAPL | 3 | 2024-01-11–2025-08-07 | 2025-08-08–2025-10-27 | 2025-10-28–2026-01-16 | 394/56/56 |
| AAPL | 4 | 2024-01-11–2025-10-27 | 2025-10-28–2026-01-16 | 2026-01-20–2026-04-09 | 450/56/56 |
| MSFT | 1 | 2024-01-11–2025-02-25 | 2025-02-26–2025-05-15 | 2025-05-16–2025-08-07 | 281/56/57 |
| MSFT | 2 | 2024-01-11–2025-05-16 | 2025-05-19–2025-08-07 | 2025-08-08–2025-10-27 | 338/56/56 |
| MSFT | 3 | 2024-01-11–2025-08-07 | 2025-08-08–2025-10-27 | 2025-10-28–2026-01-16 | 394/56/56 |
| MSFT | 4 | 2024-01-11–2025-10-27 | 2025-10-28–2026-01-16 | 2026-01-20–2026-04-09 | 450/56/56 |
| JPM | 1 | 2024-01-11–2025-02-25 | 2025-02-26–2025-05-15 | 2025-05-16–2025-08-07 | 281/56/57 |
| JPM | 2 | 2024-01-11–2025-05-16 | 2025-05-19–2025-08-07 | 2025-08-08–2025-10-27 | 338/56/56 |
| JPM | 3 | 2024-01-11–2025-08-07 | 2025-08-08–2025-10-27 | 2025-10-28–2026-01-16 | 394/56/56 |
| JPM | 4 | 2024-01-11–2025-10-27 | 2025-10-28–2026-01-16 | 2026-01-20–2026-04-09 | 450/56/56 |
| JNJ | 1 | 2024-01-11–2025-02-25 | 2025-02-26–2025-05-15 | 2025-05-16–2025-08-07 | 281/56/57 |
| JNJ | 2 | 2024-01-11–2025-05-16 | 2025-05-19–2025-08-07 | 2025-08-08–2025-10-27 | 338/56/56 |
| JNJ | 3 | 2024-01-11–2025-08-07 | 2025-08-08–2025-10-27 | 2025-10-28–2026-01-16 | 394/56/56 |
| JNJ | 4 | 2024-01-11–2025-10-27 | 2025-10-28–2026-01-16 | 2026-01-20–2026-04-09 | 450/56/56 |
| XOM | 1 | 2024-01-11–2025-02-25 | 2025-02-26–2025-05-15 | 2025-05-16–2025-08-07 | 281/56/57 |
| XOM | 2 | 2024-01-11–2025-05-16 | 2025-05-19–2025-08-07 | 2025-08-08–2025-10-27 | 338/56/56 |
| XOM | 3 | 2024-01-11–2025-08-07 | 2025-08-08–2025-10-27 | 2025-10-28–2026-01-16 | 394/56/56 |
| XOM | 4 | 2024-01-11–2025-10-27 | 2025-10-28–2026-01-16 | 2026-01-20–2026-04-09 | 450/56/56 |

## Package versions

| Package | Version |
| --- | --- |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| tensorflow | 2.21.0 |
| transformers | 5.16.1 |
| xgboost | 3.4.1 |
| yfinance | 1.7.0 |


## Walk-forward mean across folds, by ticker

### AAPL

| Model | Accuracy mean ± SD | Balanced accuracy mean ± SD | ROC-AUC mean ± SD | Brier mean ± SD | Return mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous_direction | 0.5380 ± 0.0496 | 0.5332 ± 0.0508 | 0.5332 ± 0.0508 | 0.4620 ± 0.0496 | 3.5131 ± 6.9163% |
| random_forest | 0.5157 ± 0.0276 | 0.5067 ± 0.0411 | 0.5296 ± 0.0518 | 0.2573 ± 0.0087 | 1.5607 ± 6.6626% |
| xgboost | 0.5290 ± 0.0543 | 0.5174 ± 0.0482 | 0.5182 ± 0.0598 | 0.2882 ± 0.0052 | 2.5333 ± 9.9543% |
| lstm | 0.5287 ± 0.0550 | 0.5000 ± 0.0000 | 0.4871 ± 0.0900 | 0.2545 ± 0.0149 | 5.7606 ± 10.6144% |
| hybrid | 0.5155 ± 0.0517 | 0.5002 ± 0.0388 | 0.5240 ± 0.0385 | 0.2532 ± 0.0025 | 4.2823 ± 9.4725% |

### MSFT

| Model | Accuracy mean ± SD | Balanced accuracy mean ± SD | ROC-AUC mean ± SD | Brier mean ± SD | Return mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous_direction | 0.4894 ± 0.0862 | 0.4753 ± 0.0920 | 0.4753 ± 0.0920 | 0.5106 ± 0.0862 | -3.6909 ± 3.4894% |
| random_forest | 0.5024 ± 0.1096 | 0.5382 ± 0.0907 | 0.5523 ± 0.0980 | 0.2670 ± 0.0307 | -3.8224 ± 9.7145% |
| xgboost | 0.5111 ± 0.0552 | 0.5385 ± 0.0559 | 0.5458 ± 0.0676 | 0.3053 ± 0.0401 | -2.4680 ± 8.6123% |
| lstm | 0.4536 ± 0.0793 | 0.4736 ± 0.0527 | 0.5667 ± 0.0965 | 0.2549 ± 0.0086 | -6.4096 ± 12.4869% |
| hybrid | 0.4535 ± 0.0801 | 0.4800 ± 0.0532 | 0.5289 ± 0.0867 | 0.2527 ± 0.0120 | -7.3622 ± 11.4477% |

### JPM

| Model | Accuracy mean ± SD | Balanced accuracy mean ± SD | ROC-AUC mean ± SD | Brier mean ± SD | Return mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous_direction | 0.4982 ± 0.0759 | 0.4889 ± 0.0796 | 0.4889 ± 0.0796 | 0.5018 ± 0.0759 | -1.6650 ± 3.3833% |
| random_forest | 0.4582 ± 0.0927 | 0.4617 ± 0.0728 | 0.4743 ± 0.0443 | 0.2692 ± 0.0169 | -1.6037 ± 4.8579% |
| xgboost | 0.5067 ± 0.0304 | 0.4945 ± 0.0312 | 0.4654 ± 0.0252 | 0.3030 ± 0.0222 | 0.1657 ± 4.9613% |
| lstm | 0.5555 ± 0.0387 | 0.5000 ± 0.0000 | 0.4095 ± 0.0660 | 0.2506 ± 0.0059 | 4.1891 ± 3.8787% |
| hybrid | 0.5158 ± 0.0413 | 0.4822 ± 0.0227 | 0.4661 ± 0.0571 | 0.2524 ± 0.0078 | 1.8125 ± 2.9031% |

### JNJ

| Model | Accuracy mean ± SD | Balanced accuracy mean ± SD | ROC-AUC mean ± SD | Brier mean ± SD | Return mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous_direction | 0.5333 ± 0.0374 | 0.5227 ± 0.0392 | 0.5227 ± 0.0392 | 0.4667 ± 0.0374 | 5.0019 ± 4.3710% |
| random_forest | 0.4576 ± 0.0224 | 0.4468 ± 0.0390 | 0.4260 ± 0.0823 | 0.2660 ± 0.0040 | 1.5445 ± 2.4903% |
| xgboost | 0.5377 ± 0.0342 | 0.5263 ± 0.0597 | 0.4793 ± 0.0504 | 0.2906 ± 0.0102 | 4.5851 ± 1.8190% |
| lstm | 0.5556 ± 0.0185 | 0.4960 ± 0.0080 | 0.4069 ± 0.0388 | 0.2483 ± 0.0027 | 11.9426 ± 3.9074% |
| hybrid | 0.5247 ± 0.0379 | 0.4820 ± 0.0152 | 0.4356 ± 0.0560 | 0.2520 ± 0.0019 | 7.4973 ± 4.8212% |

### XOM

| Model | Accuracy mean ± SD | Balanced accuracy mean ± SD | ROC-AUC mean ± SD | Brier mean ± SD | Return mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous_direction | 0.5554 ± 0.0727 | 0.5386 ± 0.0657 | 0.5386 ± 0.0657 | 0.4446 ± 0.0727 | 4.2027 ± 2.8480% |
| random_forest | 0.5998 ± 0.0330 | 0.6038 ± 0.0424 | 0.6059 ± 0.0231 | 0.2442 ± 0.0048 | 8.4904 ± 5.3255% |
| xgboost | 0.5821 ± 0.0330 | 0.5885 ± 0.0419 | 0.5806 ± 0.0349 | 0.2660 ± 0.0102 | 5.3227 ± 4.1380% |
| lstm | 0.5245 ± 0.0666 | 0.4813 ± 0.0375 | 0.5327 ± 0.0362 | 0.2483 ± 0.0019 | 7.0130 ± 10.3908% |
| hybrid | 0.5732 ± 0.0501 | 0.5465 ± 0.0572 | 0.5916 ± 0.0513 | 0.2440 ± 0.0048 | 6.3675 ± 3.5034% |

## Equal-weight mean of ticker-level walk-forward means

Each ticker is weighted equally after averaging its folds. This is descriptive and is not a combined portfolio.

| Model | Accuracy mean +/- SD | Balanced accuracy mean +/- SD | ROC-AUC mean +/- SD | Brier mean +/- SD | Return mean +/- SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous_direction | 0.5229 +/- 0.0279 | 0.5117 +/- 0.0281 | 0.5117 +/- 0.0281 | 0.4771 +/- 0.0279 | 1.4724 +/- 3.8916% |
| random_forest | 0.5067 +/- 0.0582 | 0.5114 +/- 0.0631 | 0.5176 +/- 0.0697 | 0.2607 +/- 0.0103 | 1.2339 +/- 4.6497% |
| xgboost | 0.5333 +/- 0.0301 | 0.5330 +/- 0.0349 | 0.5179 +/- 0.0473 | 0.2906 +/- 0.0157 | 2.0278 +/- 3.2145% |
| lstm | 0.5236 +/- 0.0417 | 0.4902 +/- 0.0120 | 0.4806 +/- 0.0719 | 0.2513 +/- 0.0032 | 4.4991 +/- 6.7530% |
| hybrid | 0.5165 +/- 0.0426 | 0.4982 +/- 0.0282 | 0.5092 +/- 0.0606 | 0.2509 +/- 0.0039 | 2.5195 +/- 5.9338% |

## Final chronological holdout, by ticker

| Ticker | Model | Accuracy | Balanced accuracy | Precision up | Recall up | F1 up | ROC-AUC | Brier | Long-or-cash return | Max drawdown | Sharpe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AAPL | previous_direction | 0.4043 | 0.3991 | 0.4545 | 0.4545 | 0.4545 | 0.3991 | 0.5957 | -2.9425% | -6.4960% | -0.2995 |
| AAPL | random_forest | 0.4823 | 0.4745 | 0.5244 | 0.5584 | 0.5409 | 0.5146 | 0.2534 | 13.7829% | -4.8228% | 1.4903 |
| AAPL | xgboost | 0.5461 | 0.5303 | 0.5684 | 0.7013 | 0.6279 | 0.5668 | 0.2567 | 17.4029% | -10.2279% | 1.6029 |
| AAPL | lstm | 0.5461 | 0.5000 | 0.5461 | 1.0000 | 0.7064 | 0.4507 | 0.2481 | 32.4087% | -12.7062% | 2.0456 |
| AAPL | hybrid | 0.5603 | 0.5288 | 0.5630 | 0.8701 | 0.6837 | 0.4748 | 0.2503 | 25.7206% | -12.3845% | 1.9448 |
| AAPL | buy_and_hold | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 32.4087% | -12.7062% | 2.0456 |
| AAPL | flat_cash | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.0000% | 0.0000% | 0.0000 |
| MSFT | previous_direction | 0.4752 | 0.4720 | 0.5132 | 0.5132 | 0.5132 | 0.4720 | 0.5248 | 4.6909% | -13.2751% | 0.4818 |
| MSFT | random_forest | 0.5532 | 0.5521 | 0.5890 | 0.5658 | 0.5772 | 0.5419 | 0.2561 | 20.4060% | -22.5995% | 1.2066 |
| MSFT | xgboost | 0.5248 | 0.5247 | 0.5634 | 0.5263 | 0.5442 | 0.5150 | 0.2874 | 29.9975% | -12.5486% | 1.7003 |
| MSFT | lstm | 0.5390 | 0.5000 | 0.5390 | 1.0000 | 0.7005 | 0.4455 | 0.2496 | 41.1598% | -23.3844% | 1.8810 |
| MSFT | hybrid | 0.4681 | 0.4531 | 0.5052 | 0.6447 | 0.5665 | 0.4917 | 0.2512 | 7.8928% | -26.7210% | 0.5718 |
| MSFT | buy_and_hold | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 41.1598% | -23.3844% | 1.8810 |
| MSFT | flat_cash | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.0000% | 0.0000% | 0.0000 |
| JPM | previous_direction | 0.4326 | 0.4221 | 0.5000 | 0.5000 | 0.5000 | 0.4221 | 0.5674 | -20.9964% | -22.4990% | -2.7528 |
| JPM | random_forest | 0.4823 | 0.4853 | 0.5522 | 0.4625 | 0.5034 | 0.4988 | 0.2556 | -3.4005% | -10.6939% | -0.3670 |
| JPM | xgboost | 0.5461 | 0.5221 | 0.5833 | 0.7000 | 0.6364 | 0.5309 | 0.2652 | 0.0986% | -9.5403% | 0.0942 |
| JPM | lstm | 0.5674 | 0.5000 | 0.5674 | 1.0000 | 0.7240 | 0.5287 | 0.2466 | 9.6948% | -9.4063% | 0.9348 |
| JPM | hybrid | 0.5461 | 0.5182 | 0.5800 | 0.7250 | 0.6444 | 0.4754 | 0.2499 | 3.1215% | -9.8368% | 0.4028 |
| JPM | buy_and_hold | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 9.6948% | -9.4063% | 0.9348 |
| JPM | flat_cash | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.0000% | 0.0000% | 0.0000 |
| JNJ | previous_direction | 0.5035 | 0.5023 | 0.5270 | 0.5270 | 0.5270 | 0.5023 | 0.4965 | 6.2799% | -10.0671% | 0.6995 |
| JNJ | random_forest | 0.5390 | 0.5382 | 0.5616 | 0.5541 | 0.5578 | 0.5490 | 0.2485 | 15.2926% | -7.9104% | 1.7173 |
| JNJ | xgboost | 0.4894 | 0.4867 | 0.5128 | 0.5405 | 0.5263 | 0.5442 | 0.2689 | 4.9071% | -11.1131% | 0.6374 |
| JNJ | lstm | 0.5248 | 0.5000 | 0.5248 | 1.0000 | 0.6884 | 0.5063 | 0.2505 | 15.6796% | -10.7101% | 1.2200 |
| JNJ | hybrid | 0.5461 | 0.5309 | 0.5439 | 0.8378 | 0.6596 | 0.5321 | 0.2469 | 13.7234% | -9.4293% | 1.2582 |
| JNJ | buy_and_hold | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 15.6796% | -10.7101% | 1.2200 |
| JNJ | flat_cash | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.0000% | 0.0000% | 0.0000 |
| XOM | previous_direction | 0.5106 | 0.5043 | 0.5641 | 0.5570 | 0.5605 | 0.5043 | 0.4894 | 0.3929% | -15.6075% | 0.1373 |
| XOM | random_forest | 0.5532 | 0.5371 | 0.5889 | 0.6709 | 0.6272 | 0.5265 | 0.2538 | 5.9383% | -10.1890% | 0.5886 |
| XOM | xgboost | 0.4965 | 0.4830 | 0.5465 | 0.5949 | 0.5697 | 0.4782 | 0.2834 | 4.9213% | -15.9477% | 0.4986 |
| XOM | lstm | 0.5603 | 0.5000 | 0.5603 | 1.0000 | 0.7182 | 0.5480 | 0.2475 | 24.2883% | -16.2965% | 1.6001 |
| XOM | hybrid | 0.5674 | 0.5202 | 0.5714 | 0.9114 | 0.7024 | 0.4996 | 0.2483 | 15.9998% | -14.5343% | 1.1657 |
| XOM | buy_and_hold | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 24.2883% | -16.2965% | 1.6001 |
| XOM | flat_cash | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.0000% | 0.0000% | 0.0000 |

## Limitations

- This optional benchmark reused the previously evaluated final-holdout dates for the additional XGBoost model. No settings were tuned against those outcomes, but these holdout results are exploratory and are not a pristine confirmatory test.
- Yahoo Finance is a provider-dependent source; values are adjusted daily bars and may be revised or unavailable. Run start/completion times and each data fingerprint are recorded; exact per-request provider response timestamps were not captured.
- This is one 3-year historical sample per ticker. The five large-cap tickers (AAPL, MSFT, JPM, JNJ, XOM) span several sectors but are a small, current-company sample with selection and survivorship bias; they do not represent a broad universe or every market regime.
- Yahoo Finance does not provide point-in-time historical news in this pipeline. FinBERT is neutral in every historical row; the live news contribution is not evaluated.
- The LSTM uses fixed configured architecture/training settings and validation early stopping; there is no nested model search. The fixed hybrid weights are a baseline, not learned weights.
- The final holdout is evaluated once in this report. Do not tune settings, weights, cutoffs, or claims against these final-holdout results.
- The final holdout window was also scored in the superseded preliminary run that exposed the one-row fold. The fold splitter was corrected based on fold-size diagnostics, not holdout metrics; no model parameters, weights, or thresholds were changed. Treat the final holdout as exploratory rather than pristine confirmatory evidence.
- Probability scores are raw model probabilities/proxies. Brier/reliability metrics assess observed quality; no calibration was fitted. Direction probability is not probability of profit.
- The 0.50 classifier cutoff is fixed, not validation-selected. Reliability bins are descriptive only; fold standard deviations are not confidence intervals or significance tests.
- Trading results are simplified close-to-close long-or-cash calculations. The configured 0.06% fee and 0.04% slippage are charged on position changes; there is no share/cash ledger or separate final liquidation charge. These are not the next-open execution backtest and omit taxes, market impact, liquidity constraints, and capital constraints.
- TensorFlow ran on CPU on native Windows; GPU support is unavailable in this setup. Prediction-function retracing warnings increased runtime but training completed. Fixed seeds are set, but exact neural reproducibility across libraries/hardware is not guaranteed.
- Per-fold aggregate outputs are saved, but per-date component predictions are not exported.
- The target is one-session direction only. There is no separate 1W/1M/3M trained evaluation, no intraday data, and no fundamentals or point-in-time corporate/news features.
- Synthetic fixture results are not included here and are not empirical market evidence.

## Machine-readable details

The adjacent `phase7_xgboost_evaluation.json` contains per-fold date ranges, predictions' aggregate metrics, component statuses, run configuration, package versions, data hashes, and errors for failed ticker fetches/evaluations.
