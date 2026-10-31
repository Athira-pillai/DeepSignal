# Phase 3 — Walk-forward HMM regime evaluation

This report describes fixed-hybrid predictions pooled together and stratified by causal HMM regime. It does not test whether adding regime as a model feature improves predictions.

## Method

- Target: next-session adjusted close above the current adjusted close.
- HMM inputs: daily return, 20-session rolling volatility, normalized high-low range, and volume change; calculated through each daily close.
- A StandardScaler and diagonal Gaussian HMM are fit on each fold's training rows only. Two- and three-state candidates are selected with later validation mean log likelihood and a parameter penalty.
- State descriptions are assigned from training-window average return and volatility; the mapping is fixed within each test fold. State IDs are fold-local.
- Test posteriors use prefixes ending at each test date. The final chronological holdout is excluded.
- HMM library: pomegranate 1.1.2; predictions use the same fixed-hybrid model outputs as the pooled baseline.

## AAPL — ok

Pooled OOS rows: 225; folds: 4; final holdout excluded: True.
Without regime stratification — accuracy 0.5156, balanced accuracy 0.5013, ROC-AUC 0.496, Brier 0.2532.

| Inferred regime | Rows | Accuracy | Balanced accuracy | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|
| bearish/high-risk | 3 | 0.6667 | None | None | 0.1931 |
| bullish/trending | 112 | 0.5268 | 0.5029 | 0.4899 | 0.2534 |
| bearish/low-volatility | 110 | 0.5 | 0.5 | 0.4932 | 0.2547 |

Price data: 2023-10-31 to 2026-10-30; SHA-256 `c9c33cc4dadf6252f41d8a389ad6241ee35abea27e3d3eff5bf427ec36b83fad`.

| Fold | Selected states | Test dates | Test regime counts |
|---:|---:|---|---|
| 1 | 2 | 2025-05-16 to 2025-08-07 | `{"bearish/high-risk": 2, "bullish/trending": 55}` |
| 2 | 3 | 2025-08-08 to 2025-10-27 | `{"bearish/high-risk": 1, "bullish/trending": 55}` |
| 3 | 2 | 2025-10-28 to 2026-01-16 | `{"bearish/low-volatility": 54, "bullish/trending": 2}` |
| 4 | 2 | 2026-01-20 to 2026-04-09 | `{"bearish/low-volatility": 56}` |

Full candidate scores, training/validation/test windows, per-fold metrics and state mappings are in the JSON report.

## Limitations

- This is one ticker and one provider-dependent three-year sample. Regimes are latent model clusters, not known market regimes or ground truth.
- The per-regime table is descriptive stratification of unchanged fixed-hybrid predictions. It is not a causal ablation, and it does not establish predictive benefit from HMM context.
- State count and state descriptions can change by fold. Small groups can make the per-regime metrics unstable.
- State probabilities are model state posteriors, not calibrated direction probabilities or probabilities of profit.
- Yahoo adjusted daily bars are provider-dependent; no intraday path, spread, market impact, taxes or funding are modeled in this classification comparison.
- This run excludes the final holdout and is exploratory; it does not constitute a later untouched confirmation experiment.
