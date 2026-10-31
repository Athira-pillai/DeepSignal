# Phase 5 - Calibration and uncertainty-aware No Trade evaluation

**Ticker:** AAPL  
**Window:** 2025-05-16 to 2026-04-09  
**OOS sessions:** 224 over 4 folds; final holdout excluded.

## Untouched test-fold calibration metrics

**Calibrated event:** Probability that the next session close is above the current session close. Calibration estimates direction only, not trading profit.

| Model score | Samples | Brier score | Reliability bins |
|---|---:|---:|---:|
| Fixed raw | 225 | 0.2532008062528514 | 3 |
| Fixed calibrated | 225 | 0.26756809505139606 | 6 |
| Adaptive raw | 225 | 0.2541663314975201 | 3 |
| Adaptive calibrated | 225 | 0.2639140651957631 | 5 |

## No Trade backtest

No Trade decisions: 37 of 225 audited OOS rows (16.4% abstained).
Backtest return: 10.1851%; max drawdown: -13.7985%; Sharpe: 0.7945; trades: 8.
No Trade retains an existing position and does not open a new position; the execution engine represents it as HOLD.

## Fold selection audit

| Fold | Calibration method | Validation fit rows | Selection Brier | No Trade margin | Dispersion limit | Test Brier (raw -> calibrated) |
|---:|---|---:|---|---:|---:|---|
| 1 | platt; platt: ok, isotonic: unavailable (insufficient_validation_samples_or_class_diversity) | 56 | platt 0.2702 | 0.0 | 0.1 | 0.2523 -> 0.2576 |
| 2 | platt; platt: ok, isotonic: unavailable (insufficient_validation_samples_or_class_diversity) | 56 | platt 0.2478 | 0.0 | 1.0 | 0.2542 -> 0.2490 |
| 3 | platt; platt: ok, isotonic: unavailable (insufficient_validation_samples_or_class_diversity) | 56 | platt 0.2545 | 0.0 | 1.0 | 0.2561 -> 0.2785 |
| 4 | platt; platt: ok, isotonic: unavailable (insufficient_validation_samples_or_class_diversity) | 56 | platt 0.2891 | 0.0 | 0.15 | 0.2503 -> 0.2853 |

## Method

- Each fold generates component probabilities on a preceding chronological validation window. Adaptive validation probabilities use only earlier rows when choosing their weights.
- Platt scaling and isotonic regression are candidates. Candidate calibrators fit on the first 60% of validation rows; validation Brier score on the later 40% selects a method; the selected calibrator is refit on the full past validation window before test inference.
- No Trade neutral-band and maximum component-dispersion thresholds are selected using the later validation slice and frozen for the next test fold. Test labels are used only for evaluation.
- Reliability uses fixed-width probability bins. The full fold-level audit, selected methods, cutoffs, raw/calibrated outputs, and data hashes are in the JSON file.
- Uncertainty is the standard deviation among available LSTM, Random Forest, and technical component probabilities. It is a disagreement estimate, not a confidence interval or probability of profit.

## Limitations

This is one ticker/window with few folds. Isotonic regression needs a larger calibration prefix than many folds provide and will be reported unavailable when evidence is insufficient. Platt scaling and the validation-selected No Trade cutoffs can overfit small validation windows. The uncertainty proxy captures component disagreement only. Historical FinBERT news is neutral/unavailable; Yahoo daily data, transaction costs, slippage and next-open fills are simplified. Calibration metrics describe direction probability, never probability of profit. The live prediction endpoint explicitly reports calibration unavailable because it does not fit a point-in-time validation calibrator at request time.
