# Codex implementation progress

## Current project status

**The original nine phases are complete.** Phase 9 generated the final evidence report; its E5 sentiment ablation is explicitly unavailable because no eligible point-in-time historical news corpus was available. The original implementation plan defines no Phase 10. Further research can resume when there is a licensed historical news archive for E5 or a new untouched confirmation window after a prospective evaluation plan is fixed.

Global safeguards are covered by implementation and tests: chronological folds, training-only preprocessing and HMM state mapping, validation-only adaptive/calibration selection, bounded normalized weights, explicit optional-component/news failures, next-open execution, costs/slippage, versioned experiment metadata, and UI timestamps that distinguish market date from retrieval time. The project has no brokerage connection or order execution. Full backend suite: 62 passed; Phase 8 Vite production build: passed; Phase 9 report JSON and SVG validation: passed; `git diff --check`: passed.

## Phase 0 â€” Repository inspection, safe checkpoint and baseline

**Status: completed.** Repository inspection and an offline baseline were completed. The clean working tree was checkpointed on `feature/deepsignal-phase-zero`. No existing data or model artifacts were modified.

### Repository findings

- React/Vite frontend and FastAPI backend, mapped in [docs/DeepSignal_Project_Guide.md](docs/DeepSignal_Project_Guide.md).
- Live hybrid: fixed 40/40/10/10 blend; target is next-session close above current close; signal thresholds are documented in [research/baseline/PHASE_0_BASELINE.md](research/baseline/PHASE_0_BASELINE.md).
- Existing evaluator has a single chronological holdout and neutral historical sentiment. The stored evaluation artifact contains only previous-direction and RF metrics, not the current evaluator's LSTM/hybrid outputs.
- Existing `/api/backtest/{symbol}` runs a technical momentum/RSI long-only strategy using next-open execution and costs, not live hybrid predictions. The persisted report keeps a `hybrid` compatibility alias with identical technical-strategy metrics.
- Model artifacts are stored locally in `backend/model_artifacts/`; optional Supabase is separate persistence for records.

### Files changed

- `CODEX_PROGRESS.md` â€” phase status, findings and checks.
- `docs/DeepSignal_Project_Guide.md` â€” actual code-path map and discrepancies.
- `research/baseline/PHASE_0_BASELINE.md` â€” summarized preexisting baseline artifacts, fingerprints, metrics and assumptions.

### Commands and results

- `git status --short --branch` â€” clean before work, `main...origin/main`; remains clean aside from the three new documentation files.
- `git switch -c feature/deepsignal-phase-zero` â€” first sandboxed attempt could not write `.git`; rerun with Git metadata write authorization succeeded.
- `\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v` â€” **25 tests passed**.
- `npm.cmd run build` in `frontend/` â€” **passed**, Vite 6.4.3 production build. First sandboxed invocation failed due filesystem access to Vite/esbuild above the workspace; approved retry completed.
- `Invoke-RestMethod http://127.0.0.1:8000/api/health` â€” returned `status: ok`. A second Uvicorn process could not bind because port 8000 was already occupied; no process was stopped.
- No lint or type-check command is configured.
- No new evaluation/model fetch was run. The saved AAPL artifacts are dated 2026-10-24; baseline values and report limitations are recorded in `research/baseline/PHASE_0_BASELINE.md`.

### Acceptance checklist

- [x] Repository and implementation paths inspected.
- [x] Preexisting working tree confirmed clean; existing files/artifacts preserved.
- [x] Feature branch created: `feature/deepsignal-phase-zero`.
- [x] Existing backend suite passes.
- [x] Frontend production build passes.
- [x] Running API health endpoint responds; no second server could bind on occupied port 8000.
- [x] Existing fixed hybrid and technical backtest are documented separately.
- [x] Baseline data hashes, dates, model metrics and backtest assumptions saved.
- [x] Implementation map and phase progress note added.

### Limitations and next phase

The baseline is based on preexisting AAPL artifacts, not a fresh Yahoo Finance download or rerun. The stored evaluation run does not include LSTM or hybrid metrics despite those models being implemented at the time. The Phase 1 evaluator and limits are documented below.

## Phase 1 â€” Leakage-safe walk-forward evaluation

**Status: completed; findings are mixed and do not establish model improvement.** The evaluator now generates expanding folds in the development window and reserves the final chronological 20% as a separate holdout. A five-ticker Yahoo Finance evaluation is saved in `research/real_data/`. A first run revealed a one-row partial fold; fold sizing was corrected to make test windows balanced, then all five tickers were rerun.

### Implementation

- `backend/evaluation.py` replaces the single-split evaluator with expanding train/validation/test folds and one untouched final holdout. Existing `models` field remains the final-holdout model table for compatibility; new fields provide per-fold metrics and aggregates.
- Models compared on matching dates: previous-direction baseline, Random Forest, LSTM, fixed-weight hybrid; optional XGBoost is controlled by `EVALUATION_INCLUDE_XGBOOST` and disabled by default. Historical sentiment stays neutral because point-in-time news is unavailable.
- Metrics include accuracy, balanced accuracy, precision, recall, F1, confusion matrix, class counts, ROC-AUC when both classes exist, Brier score, reliability bins, and the existing simplified trading metrics.
- LSTM scaling is fit on training rows per fold; the validation interval is used for early stopping. Target is next-session close above current-session close. Test labels are used only after predictions for scoring.
- `GET /api/evaluate?symbols=AAPL,MSFT,NVDA` provides a configurable comma-separated ticker run; defaults come from `EVALUATION_SYMBOLS`.
- `tools/run_walk_forward_fixture.py` creates a reproducible synthetic smoke report at `research/walk_forward/synthetic_fixture_report.json`. Its LSTM output is neutral-stubbed by design; its synthetic metrics are not market-performance evidence.
- `tools/run_real_evaluation.py` fetches 3 years of Yahoo Finance adjusted daily data, evaluates each ticker, and writes a standalone report without appending to the existing `backend/evaluation_runs.json`.
- Real-data report: [research/real_data/PHASE_1_REAL_DATA_EVALUATION.md](research/real_data/PHASE_1_REAL_DATA_EVALUATION.md); machine-readable output: `research/real_data/phase1_real_data_evaluation.json`.
- Detailed method and evidence limits: [research/walk_forward/PHASE_1_REPORT.md](research/walk_forward/PHASE_1_REPORT.md).

### Files changed

- `backend/evaluation.py` â€” fold generator, train-only scaling, optional XGBoost, expanded metrics, fold aggregation, report metadata.
- `backend/main.py` â€” multi-symbol evaluation endpoint and environment-driven optional XGBoost.
- `backend/tests/test_hybrid.py`, `backend/tests/test_validation.py` â€” temporal, metrics, scaler, optional-model and route coverage.
- `backend/.env.example`, `README.md` â€” evaluation settings and usage.
- `tools/run_walk_forward_fixture.py`, `research/walk_forward/` â€” reproducible synthetic smoke run and report.
- `docs/DeepSignal_Project_Guide.md` â€” updated path map and branch status.
- `tools/run_real_evaluation.py`, `research/real_data/` â€” real-data experiment driver, Markdown report, JSON metrics and data fingerprints.
- `CODEX_PROGRESS.md` â€” this phase record.

### Validation commands and results

- `\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v` â€” **32 tests passed** after the final edits.
- `\.venv\Scripts\python.exe -m py_compile backend\evaluation.py backend\main.py backend\tests\test_hybrid.py backend\tests\test_validation.py tools\run_walk_forward_fixture.py tools\run_real_evaluation.py` â€” passed after the final edits.
- `\.venv\Scripts\python.exe tools\run_walk_forward_fixture.py` â€” returned `status=ok`, four expanding folds, a 150-row final holdout, shared model test dates, and fingerprint `daa875fa32bed04039d147dfd66be993505e59a6e4abfa38abd67ecc4f124588`. Source is explicitly marked synthetic.
- Direct one-epoch LSTM smoke check after the final Keras input-layer change â€” returned `trained`, five finite probabilities. No external data was fetched.
- `\.venv\Scripts\python.exe tools\run_real_evaluation.py --symbols AAPL,MSFT,JPM,JNJ,XOM --period 3y` (network-enabled run) â€” all five tickers succeeded; each had 753 raw rows, 703 labeled rows, four balanced folds (56â€“57 test rows), and 141 final-holdout rows from 2026-04-10 to 2026-10-29. Per-ticker SHA-256 hashes and dates are in the JSON report.
- Walk-forward equal-ticker mean accuracy / balanced accuracy / ROC-AUC: previous direction `0.5229 / 0.5117 / 0.5117`; RF `0.5076 / 0.5123 / 0.5177`; LSTM `0.5236 / 0.4902 / 0.4806`; fixed hybrid `0.5165 / 0.4982 / 0.5094`. These are means of per-ticker fold means, not a portfolio or a significance test.
- Final-holdout equal-ticker mean accuracy: previous direction `0.4652`, RF `0.5192`, LSTM `0.5475`, hybrid `0.5376`. Hybrid mean AUC was `0.4944`, versus RF `0.5258`; results conflict across metrics and do not support a general improvement claim.
- TensorFlow emitted CPU/GPU and prediction retracing warnings. LSTM fit succeeded on each ticker; XGBoost was disabled by configuration.

### Acceptance checklist

- [x] Expanding chronological fold generator and reserved final holdout.
- [x] Target alignment and train-only scaler have unit coverage.
- [x] Models are evaluated on matching dates within each fold; probability and classification metrics include single-class handling.
- [x] Missing/disabled XGBoost is explicit; optional comparison can be enabled without adding packages.
- [x] Multi-ticker coverage is configurable and API validation is tested.
- [x] Reproducible multi-fold report generated on a clearly labeled synthetic fixture.
- [x] Real Yahoo Finance multi-ticker report generated with multiple sectors, fingerprints, fold/holdout dates, metrics and limitations documented.
- [x] Final full-suite and syntax checks after last edits.

### Limitations and next step

The dataset is one provider-dependent three-year window over five selected large-cap tickers; historical news is neutral, XGBoost is disabled, probabilities are uncalibrated, and trading metrics are simplified close-to-close simulations. Fold standard deviations are descriptive, not confidence intervals. The LSTM predicts only the up class at the 0.50 cutoff on each final holdout (balanced accuracy 0.50). The same final holdout was scored in the superseded first run and the corrected run; the correction was prompted by a one-row fold and did not change model settings, but the final results should be treated as exploratory rather than pristine confirmatory evidence. These and other limitations are recorded in the real-data report. Phase 1 is complete.

## Phase 2 â€” Walk-forward hybrid-signal backtest

**Status: completed; historical result is exploratory and does not establish future profitability.** Added a separate hybrid backtest workflow and endpoint. The existing `/api/backtest/{symbol}` remains the technical-strategy backtest. The new `/api/backtest/{symbol}/hybrid` uses only model predictions generated on expanding development test folds and excludes the final 20% holdout.

### Implementation

- `backend/hybrid_backtesting.py` obtains per-fold LSTM, Random Forest, and technical probabilities from the Phase 1 evaluator, combines them through the production fixed-weight hybrid, and compares hybrid/LSTM-only/RF-only strategies on matching out-of-sample dates.
- The last test-date signal is retained in the audit and excluded from orders because no next open is available inside the test window. All executed signal trades follow after-close decision / next-open execution, long-only sizing, modeled costs and slippage, and terminal close liquidation.
- `backend/backtesting.py` trade rows now include before/after share positions plus signal and execution timestamps. `backend/hybrid.py` exposes the same production threshold mapper for component-only comparisons.
- `GET /api/backtest/{symbol}/hybrid` persists through the existing experiment storage. The Backtest page switches between technical and hybrid modes and shows the three model strategy results.
- `tools/run_hybrid_backtest.py` reproducibly fetches a public three-year Yahoo window and writes a standalone report. Markdown method/results/limitations and JSON signal audit are in `research/real_data/PHASE_2_HYBRID_BACKTEST.md` and `research/real_data/phase2_hybrid_backtest.json`.

### Real-data result

One AAPL run: 703 labeled rows; four expanding folds with LSTM trained in each; the executable window was 2025-05-16 through 2026-04-09 (225 price rows, 224 measured sessions). Fixed hybrid returned **25.1038%**, maximum drawdown **âˆ’16.1417%**, Sharpe **1.2878**, 12 trades. Buy-and-hold returned **22.7918%**, drawdown **âˆ’13.7985%**, Sharpe **1.1746**. LSTM-only returned 25.4199%; RF-only 19.3772%. The hybrid beat the benchmark return in this sample but had a deeper drawdown; one ticker/window cannot establish a reliable ranking.

Limitations include neutral historical sentiment, fixed heuristic and uncalibrated thresholds, one symbol/window, four folds, provider-adjusted daily bars, simplified fees/slippage/fractional fills, and no impact/spread/tax/funding models. Full assumptions, data fingerprints, results and limitations are in the report.

### Validation

- `\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v` â€” **34 tests passed**.
- `\.venv\Scripts\python.exe -m py_compile backend\hybrid.py backend\backtesting.py backend\hybrid_backtesting.py backend\main.py backend\tests\test_hybrid_backtesting.py tools\run_hybrid_backtest.py` â€” passed.
- `npm.cmd run build` â€” Vite production build passed (2,211 modules).
- `git diff --check` â€” passed.
- `\.venv\Scripts\python.exe tools\run_hybrid_backtest.py --symbol AAPL --period 3y` â€” real-data run completed; full report and signals saved. TensorFlow logged CPU-only/retracing warnings; each fold's LSTM fit completed.

### Acceptance checklist

- [x] Separate hybrid route preserves technical backtest route.
- [x] Walk-forward OOS component probabilities use the same production hybrid weighting and signal thresholds.
- [x] Hybrid, LSTM-only and RF-only share test dates; final holdout excluded.
- [x] Next-open execution, costs, slippage, long-only state, timestamps and final liquidation assumptions are recorded.
- [x] Reproducible real-data report, data hashes, session-level signal audit, and explicit limitations saved.
- [x] Backend tests, syntax check, frontend build, and diff check pass.

Phase 2 is complete.

## Phase 3 â€” Hidden Markov market-regime detection

**Status: completed.** Added a causal daily Gaussian HMM regime estimator, optional-safe API output, a dashboard panel, and a descriptive walk-forward comparison. Existing model/hybrid outputs remain unchanged; this phase does not adapt ensemble weights.

### Implementation

- `backend/regime_detection.py` builds daily return, 20-session volatility, normalized high-low range, and volume-change features. Features are calculated through the session close; each fold's `StandardScaler` and HMM are fit on training rows only.
- Uses pinned `pomegranate==1.1.2` with the existing PyTorch dependency. It tests two- and three-state diagonal Gaussian HMMs, selects using later validation mean log likelihood with a complexity penalty, and returns an explicit fallback if no candidate converges.
- State descriptions derive from training-window mean return and volatility and remain fixed through that fold's test window. State IDs and mappings are reported by fold and are not compared as stable IDs across retraining.
- Each posterior uses only an observation prefix ending at its inference date. `GET /api/regime/{symbol}` returns regime/probabilities, recent regime history, model and package versions, selected state count and scores, training/validation windows, data hash, and fallback reason where relevant.
- Added a Dashboard Market Regime panel that shows daily cutoff separately from retrieval time, labels probabilities as latent-state probabilities (not return/profit probabilities), includes recent history, and handles unavailable data/model results.
- `evaluate_regime_walk_forward` compares pooled fixed-hybrid metrics with the same unchanged predictions stratified by fold-local HMM state descriptions. `tools/run_regime_evaluation.py` writes JSON and Markdown; `--report-only` regenerates Markdown from saved JSON.
- Real-data report and audit: `research/real_data/phase3_regime_evaluation.md` and `research/real_data/phase3_regime_evaluation.json`.

### Real-data result

AAPL, 3 years of Yahoo adjusted daily OHLCV, 753 raw rows, four expanding folds and 225 shared OOS rows; the final 20% holdout was excluded. State-count selection chose two or three states depending on fold. The pooled fixed hybrid had accuracy **0.5156**, balanced accuracy **0.5013**, ROC-AUC **0.4960**, and Brier score **0.2532**. After applying training-return/volatility label mapping, grouped accuracy was **0.6667** (bearish/high-risk, 3 rows), **0.5268** (bullish/trending, 112 rows), and **0.5000** (bearish/low-volatility, 110 rows). These are descriptive groups of identical model predictions, not an ablation and not evidence that regime awareness improves prediction. The three-row high-risk group is too small for a meaningful estimate.

### Validation

- `\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v` â€” **43 tests passed**.
- Focused HMM and route tests cover causal feature rows, train-only scaling, validation state selection, state mapping from training statistics, filtered-prefix inference, explicit short/missing-data/dependency/convergence failures, route fallback, and walk-forward pooled/stratified outputs.
- `\.venv\Scripts\python.exe -m py_compile backend\regime_detection.py backend\main.py backend\tests\test_regime_detection.py backend\tests\test_routes.py tools\run_regime_evaluation.py` â€” passed.
- `npm.cmd run build` â€” Vite production build passed (2,211 modules).
- `git diff --check` â€” passed.
- `\.venv\Scripts\python.exe tools\run_regime_evaluation.py --symbols AAPL --period 3y` â€” completed; report saved with data hash, per-fold state counts and selection scores. TensorFlow emitted CPU-only warnings during the unchanged hybrid evaluation.
- `\.venv\Scripts\python.exe tools\run_regime_evaluation.py --report-only` â€” regenerated Markdown from the saved JSON report.

### Acceptance checklist

- [x] HMM state count chosen using validation evidence; train scaler/model and state descriptions use training data only.
- [x] Current regime inference is causal through the stated cutoff and returns state probabilities and model/data provenance.
- [x] Fold-local mappings remain fixed through each fold; state IDs are not treated as cross-fold identities.
- [x] Short history, invalid data, unavailable dependency/provider, and failed convergence have explicit fallback results; existing app routes remain usable.
- [x] Tested API response exposes the regime data; dashboard panel is optional-safe and distinguishes daily cutoff/retrieval time.
- [x] Walk-forward pooled and regime-context metrics run on matching test dates with final holdout excluded.
- [x] Backend suite, syntax check, frontend build, real-data report, and diff check pass.

### Limitations and next phase

HMM states are latent Gaussian clusters rather than ground-truth regimes. Their descriptions depend on training-window average return/volatility and can change by fold. State posterior values are not calibrated probabilities of direction or profit. Validation log-likelihood and a simple parameter penalty select between only two and three states; this is a compact research heuristic. The one-ticker report is descriptive, small regime groups are unstable, daily Yahoo data are provider-dependent, and no improvement claim is supported. The `pomegranate` dependency is pinned to 1.1.2 and Torch is reused from the existing requirements. Phase 3 is complete; Phase 4 adaptive weighting is complete; see the Phase 4 entry below.


## Phase 4 - Adaptive ensemble weighting

**Status: completed; fixed weights remain the default.** Implemented an opt-in adaptive hybrid strategy alongside the preserved 40/40/10/10 fixed baseline. `/api/backtest/{symbol}/hybrid` compares both strategies on identical expanding walk-forward development dates and selects the fixed or adaptive result through `mode` (or `HYBRID_BACKTEST_MODE`, which defaults to `fixed`). The Backtest page exposes the mode switch and both sets of metrics.

### Implementation

- `backend/adaptive_ensemble.py` scores available LSTM, Random Forest, and technical component predictions using Brier score on the preceding fold's out-of-sample validation window. It combines validation skill with renormalized fixed priors and projects weights onto configurable nonnegative lower/upper bounds summing to one.
- Regime-specific evidence is used only when every selected component meets the configured regime sample minimum; otherwise overall validation evidence is used with an explicit fallback audit. Weights remain frozen throughout the following test fold. Test outcomes are not passed to weight selection.
- Historical FinBERT/news is unavailable; adaptive weights exclude it and renormalize available components. The fixed baseline preserves its existing neutral sentiment contribution. Missing LSTM/evidence and invalid settings report explicit fallback/status details.
- Each signal audit includes component probabilities, selected regime, adaptive weights, validation metrics and window, component availability, method version, and fallback reason. Both strategies use the same signal thresholds, backtesting engine, execution assumptions, dates, costs, and baselines.
- Configuration is documented in `backend/.env.example`. The reproducible runner is `tools/run_adaptive_evaluation.py`; the real-data report and full machine-readable audit are `research/real_data/phase4_adaptive_evaluation.md` and `research/real_data/phase4_adaptive_evaluation.json`.

### Real-data result

One AAPL three-year Yahoo adjusted daily OHLCV run produced four walk-forward folds and 225 audited OOS signal rows. Both strategies traded on the same 224 executable sessions, 2025-05-16 through 2026-04-09; the final holdout was excluded. Starting at $100,000, fixed hybrid ended at $125,103.72 (25.1037% return, -16.1417% maximum drawdown, Sharpe 1.2878, 12 trades). Adaptive ended at $114,280.18 (14.2802% return, -18.7356% maximum drawdown, Sharpe 0.8238, 16 trades). Buy-and-hold returned 22.7918% with -13.7985% maximum drawdown. This exploratory single-symbol result does not establish future profitability, but does not justify replacing the fixed default.

### Validation

- `\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v` - **49 tests passed**.
- Focused adaptive/hybrid/route coverage verifies simplex bounds and sum, validation-only evidence, regime fallback, unavailable components, matching OOS dates, selected mode, and explicit adaptive route behavior.
- `\.venv\Scripts\python.exe -m py_compile backend\adaptive_ensemble.py backend\hybrid_backtesting.py backend\main.py backend\tests\test_adaptive_ensemble.py backend\tests\test_hybrid_backtesting.py backend\tests\test_routes.py tools\run_adaptive_evaluation.py` - passed.
- `npm.cmd run build` - Vite production build passed (2,211 modules).
- `git diff --check` - passed.
- `\.venv\Scripts\python.exe tools\run_adaptive_evaluation.py --symbol AAPL --period 3y` - completed; JSON/Markdown report saved. TensorFlow used CPU and emitted prediction-retracing warnings; LSTM training succeeded on all four folds.

### Acceptance checklist

- [x] Fixed 40/40/10/10 baseline is unchanged and remains the default.
- [x] Adaptive weights use past validation OOS predictions/outcomes only and remain frozen per test fold.
- [x] Optional HMM regime context, validation evidence minimums, component availability, bounded normalized weights, and explicit fallback audit are implemented.
- [x] Historical sentiment is treated as unavailable for adaptive weighting and documented.
- [x] Fixed and adaptive strategies share dates, backtester, execution/cost settings, and baselines; final holdout is excluded.
- [x] API mode/configuration, frontend selector, reproducible real-data report, and limitations are documented.
- [x] Backend tests, Python syntax checks, frontend production build, and diff check pass.

### Limitations

The report covers one ticker and one provider-dependent window with four validation/test folds; it is not a significance test or evidence of generalization. Brier skill and latent HMM descriptions can be noisy, regime-specific samples are sparse, and this rule-based weighting heuristic is not a calibrated meta-model. Point-in-time news is absent. Daily adjusted bars, next-open fills, fractional sizing, constant fees/slippage, and long-only positions simplify spreads, market impact, partial fills, taxes, and funding costs. Results should be independently confirmed across symbols and a later time window before changing the fixed default.

Phase 4 is complete. Phase 5 is complete; see the Phase 5 entry below.


## Phase 5 - Probability calibration and uncertainty-aware No Trade

**Status: completed; calibrated probabilities remain experimental and are not used by live predictions.** Added fold-local Platt/isotonic calibration selection, validation-selected abstention cutoffs, raw/calibrated score and uncertainty audit fields, a No Trade walk-forward strategy, and explicit live API/UI probability-quality status.

### Implementation

- `backend/calibration.py` implements Platt scaling and isotonic regression; candidates require at least 20 and 40 fit samples, respectively, and both classes. Each preceding validation window is split chronologically: candidate calibrators fit on the first 60%, later 40% selects by Brier score, and the chosen method is refit on all prior validation rows before test inference.
- Adaptive validation predictions use weights selected only from rows earlier than each prediction. Per-fold No Trade neutral-band and maximum component-disagreement thresholds are selected on the later validation slice at a minimum 35% active coverage, then frozen for that fold's test rows. Test labels are used only for reported metrics.
- No Trade is defined as HOLD: do not open a new position; retain any existing position. This matches the existing long-only execution engine.
- Walk-forward audits preserve raw fixed/adaptive probabilities, calibrated probabilities, chosen method, validation Brier scores, threshold source/values, uncertainty dispersion, trading decision, reason, target, and availability. Reports include untouched test-fold Brier/reliability results and a separate No Trade backtest.
- Live `/api/predict` returns the raw next-session direction score, `calibrated_direction_probability_up: null` with an explicit unavailable reason, finite component-dispersion uncertainty, and its raw heuristic trading decision. The UI labels the score as uncalibrated and distinguishes it from uncertainty and the decision.
- `tools/run_calibration_evaluation.py` reproduces the report and supports `--report-only`. Results and audit: `research/real_data/phase5_calibration_evaluation.md` and `.json`.

### Real-data result

AAPL, Yahoo adjusted daily OHLCV, 3-year request; four expanding development folds, 225 audited OOS rows (224 executable sessions from 2025-05-16 through 2026-04-09); final holdout excluded. Pooled fixed Brier score was **0.2532 raw** and **0.2676 calibrated**. Adaptive was **0.2542 raw** and **0.2639 calibrated**. Calibration improved the fixed score in one fold and worsened it in three; it did not improve pooled Brier. All folds selected Platt; isotonic was explicitly unavailable because the 33-row calibration prefixes did not meet its 40-row minimum.

The validation-selected No Trade strategy abstained on 37/225 audit rows (16.4%) and returned **10.1851%**, max drawdown **-13.7985%**, Sharpe **0.7945**, and 8 trades over the same 224 executable sessions. This is one exploratory ticker/window, not evidence of robust benefit.

### Validation

- `\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v` - **55 tests passed**.
- Calibration tests cover Platt and isotonic fitting, chronological selection, insufficient-data fallback, finite outputs, reliability-bin accounting, validation-only threshold selection, and No Trade status.
- Hybrid backtest tests cover identical walk-forward dates, final holdout exclusion, raw/calibrated evidence, finite uncertainty, and No Trade retaining an existing position through HOLD semantics.
- `\.venv\Scripts\python.exe -m py_compile backend\calibration.py backend\hybrid_backtesting.py backend\main.py backend\tests\test_calibration.py backend\tests\test_hybrid_backtesting.py backend\tests\test_validation.py tools\run_calibration_evaluation.py` - passed.
- `npm.cmd run build` - Vite production build passed (2,211 modules).
- `git diff --check` - passed.
- `\.venv\Scripts\python.exe tools\run_calibration_evaluation.py --symbol AAPL --period 3y` - completed; report and machine-readable audit saved. TensorFlow used CPU and emitted prediction retracing warnings; all fold model runs completed.
- `\.venv\Scripts\python.exe tools\run_calibration_evaluation.py --report-only` - regenerated Markdown from saved JSON.

### Acceptance checklist

- [x] Raw component/ensemble outputs remain available; experiment audits include raw and calibrated direction probabilities.
- [x] Platt and isotonic implementations use validation-only fit data; method selection uses a later chronological validation slice; test labels are never an input to fitting or threshold selection.
- [x] Untouched test folds report Brier and fixed-width reliability bins for raw and calibrated fixed/adaptive scores.
- [x] Finite uncertainty is explicitly labeled as component disagreement, not a confidence interval.
- [x] No Trade cutoffs are validation-selected/frozen; the rule is exercised through the shared next-open backtester.
- [x] Insufficient evidence produces explicit unavailable statuses and leaves raw baseline behavior intact.
- [x] Live UI/API distinguish raw score, unavailable calibrated probability, uncertainty, and trading decision.
- [x] Backend tests, Python syntax, frontend build, real-data report, report regeneration, and diff check pass.

### Limitations

The calibration window is short and small; test results show worse pooled Brier after Platt scaling. Isotonic could not meet its minimum sample count on any fold. The calibration fit's post-selection full-validation refit can be unstable on small samples. Component dispersion captures disagreement only, not all predictive uncertainty. No Trade thresholds optimize validation directional accuracy at a coverage floor and may overfit. Historical news is unavailable, and the report covers one ticker/window with provider-dependent daily bars and simplified execution. Live prediction intentionally does not claim a calibrated probability until a leakage-safe point-in-time calibration path is available. No calibration improvement claim is supported.

Phase 5 is complete. Phase 6 is complete under the unavailable-history branch; no sentiment ablation was run.


## Phase 6 - Historical news alignment (conditional)

**Status: completed under the unavailable-history branch; no historical sentiment ablation was run.** No permitted/reliable point-in-time corpus for the existing test dates is available in the repository. Current headlines were not substituted into historical rows.

### Provider and data findings

- The application currently uses Yahoo search, yfinance ticker news, and optional NewsAPI `/v2/everything`. Provider results include publication-time fields, but search results are not proof of a complete archive or first-public-availability timestamp.
- NewsAPI's current pricing page documents Developer search up to one month and paid Business/Advanced search up to five years. Its official terms restrict republishing third-party content and building a competing news database; the active account plan and archive retention rights were not verified. A configured key is not evidence of the necessary plan or data rights.
- Yahoo results in this project are current search/ticker results; no historical news query or archive is implemented. Yahoo terms restrict using its content to create a competing database/archive.
- No local article archive was found. Therefore historical coverage, missing-news dates, and article sample size are unknown (not zero news). The report explicitly distinguishes zero supplied/evaluated articles from zero historical news.
- Primary provider sources reviewed: `https://newsapi.org/docs/endpoints/everything`, `https://newsapi.org/pricing`, `https://newsapi.org/terms`, and `https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html`.

### Implementation

- `backend/news_alignment.py` enforces timezone-aware publication timestamps and `published_at <= cutoff`, ticker/query relevance, stable article IDs, canonical URL deduplication, and explicit rejection counters.
- Historical daily alignment uses the 16:00 `America/New_York` close converted to UTC, with DST-aware handling. It returns per-session article provenance, coverage, and missing dates when an authorized archive is supplied to the helper.
- Live `get_news` uses a request-time cutoff, drops missing/ambiguous or later articles and irrelevant query results, and retains stable references/provenance. Prediction responses expose alignment counts; the UI displays publication times, links, and rejected-item counts. This strict live path does not create historical coverage.
- `research/real_data/phase6_news_alignment.md` and `.json` record provider terms/access findings, implementation, unknown coverage, and the explicit no-ablation conclusion. `tools/run_news_alignment_report.py` reproduces the report without contacting a provider or printing credentials.

### Validation

- `\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v` - **62 tests passed**.
- New tests cover timestamp parsing, exact/as-of cutoff behavior, future/missing/ambiguous/irrelevant articles, canonical deduplication, DST-aware session assignment, missing-date coverage, no fabricated coverage, and the live news filter.
- `\.venv\Scripts\python.exe -m py_compile backend\news_alignment.py backend\main.py backend\tests\test_news_alignment.py backend\tests\test_validation.py tools\run_news_alignment_report.py` - passed.
- `npm.cmd run build` - Vite production build passed (2,211 modules).
- `git diff --check` - passed.
- `\.venv\Scripts\python.exe tools\run_news_alignment_report.py` - generated the conditional availability/limitations report; no news provider request was made.

### Acceptance checklist

- [x] Article timestamps are parsed strictly; unknown timezones and post-cutoff articles cannot enter live or historical-alignment inputs.
- [x] Duplicate references and non-relevant tickers/query results are filtered; stable source references and publication times are retained.
- [x] Historical session assignment uses an explicit NYSE-close cutoff with DST tests; helper reports coverage and missing dates when data exist.
- [x] Current providers, archive availability, access limits, and terms were inspected and documented.
- [x] The report states coverage/missingness as unknown when no corpus exists; it does not fabricate coverage or substitute current news.
- [x] No sentiment ablation or historical FinBERT value claim is made without eligible point-in-time data.
- [x] Existing live behavior remains optional-safe; backend tests, syntax, frontend build, and diff checks pass.

### Limitations and next phase

The NewsAPI account plan is not verified; Developer-plan history is too short for the existing 2025-2026 OOS window, while paid access and archival use require plan/terms confirmation. Yahoo historical-news completeness and retention rights are unverified. Timestamps are publisher-reported, not a guarantee of first public availability; ticker/query matching may still have false positives. No historical coverage or sentiment incremental-value estimate is possible from current data. Phase 6 is complete under its conditional unavailable-data path. Phase 7 optional-model comparison is complete; Phase 8 API/UI and Phase 9 final research experiments have not started.


## Phase 7 - Optional model comparison

**Status: completed.** The original Phase 7 says optional models should only be run when data/compute and a defined fair experiment are available. XGBoost was already implemented as an optional tabular benchmark, installed locally, and covered by the shared evaluator; no new architecture or dependency was added. TFT/GNN were not added because the available data/pipeline do not justify them.

### Experiment and result

- Reused the existing Phase 1 leakage-aware evaluator on AAPL, MSFT, JPM, JNJ, and XOM; 3-year Yahoo daily bars, 753 source rows per ticker, four expanding development folds, 225 shared OOS test rows per ticker, and a separate 141-row final chronological holdout. XGBoost trained in all folds. Previous-direction, RF, LSTM, and fixed hybrid remain in the same report on the same model test dates.
- Equal-ticker mean fold accuracy/balanced accuracy/AUC/Brier: XGBoost **0.5333 / 0.5330 / 0.5179 / 0.2906**; RF **0.5067 / 0.5114 / 0.5176 / 0.2607**; LSTM **0.5236 / 0.4902 / 0.4806 / 0.2513**; fixed hybrid **0.5165 / 0.4982 / 0.5092 / 0.2509**. XGBoost had the highest mean accuracy and balanced accuracy in this sample, while its Brier score was materially worse; AUC was nearly the same as RF. This does not establish general superiority or statistical significance.
- The Phase 1 final-holdout dates were reused for this additional optional model comparison. No parameters were tuned from those outcomes, but holdout metrics are explicitly labeled exploratory and are not pristine confirmatory evidence.
- Standalone report and full fold/data/configuration audit: `research/real_data/phase7_xgboost_evaluation.md` and `.json`. The prior Phase 1 report was kept as a separate file. `tools/run_real_evaluation.py` now supports a custom output path and `--report-only` for reproducible Phase 7 report regeneration.

### Validation

- `\.venv\Scripts\python.exe tools\run_real_evaluation.py --symbols AAPL,MSFT,JPM,JNJ,XOM --period 3y --include-xgboost --output research\real_data\phase7_xgboost_evaluation.json` - completed; five tickers `ok`, four folds each, XGBoost trained throughout.
- `\.venv\Scripts\python.exe tools\run_real_evaluation.py --output research\real_data\phase7_xgboost_evaluation.json --report-only` - regenerated Markdown and persisted exploratory holdout caveat.
- XGBoost version `3.4.1`; no dependency was installed or upgraded.
- Full suite and syntax/diff checks are rerun at the end of Phase 7; no frontend code changed in this phase.

### Acceptance checklist

- [x] Optional-model gate assessed against data, compute, package availability, and a defined role.
- [x] Existing XGBoost benchmark used; no model was added only to increase the model count.
- [x] XGBoost and baseline classifiers use the shared train/validation/test split generator and identical test dates/labels per ticker/fold.
- [x] Multi-ticker, multi-sector report includes aggregate and per-ticker/fold metrics, hashes, configurations, package versions, and explicit limitations.
- [x] Negative metric evidence and holdout reuse are reported without superiority claims.
- [x] Existing model/evaluation report was preserved; new output is saved separately.

### Limitations and next phase

Five selected large-cap tickers and one recent three-year provider window are not a broad universe. Fold variability is descriptive, not a confidence interval. The final holdout dates had already been evaluated in Phase 1 and are reused here; treat them as exploratory. XGBoost's higher accuracy coexists with poorer Brier score, and its ROC-AUC advantage over RF is negligible. The dataset uses neutral historical sentiment; no tuning, calibration, significance test, or new model was run. Phase 7 is complete. Phase 8 (API/persistence/frontend integration audit) and Phase 9 (final experiment matrix/report) have not started.

## Phase 8 - API, persistence, and frontend integration

**Status: implemented.** Existing response fields remain intact. Live prediction responses now explicitly state that calibrated probability and No Trade abstention are unavailable because no live validation calibrator/threshold is fitted; the raw heuristic action remains labeled as such. Experiment persistence continues to store the full versioned report, source hash, date periods, configuration, versions, and assumptions; hosted storage stores experiment records, not model binaries.

The hybrid backtest response adds fixed/adaptive strategy metrics for each shared walk-forward fold. Each fold uses matching dates and assumptions, resets starting capital, and omits its final unexecutable signal; the existing continuous aggregate remains separately available. The frontend now surfaces live No Trade availability/reason, an adaptive weight audit snapshot with regime/scope/availability, and per-fold fixed/adaptive returns and Sharpe. Existing Market Regime, model contribution, calibration, uncertainty, timestamp, loading, and unavailable/error displays were retained.

Limitations: live calibration/No Trade controls remain unavailable; historical sentiment remains neutral; market data is daily Yahoo OHLCV and not tick-level real-time. Per-fold metrics restart capital and are not directly additive to the continuous aggregate. Phase 9 is complete with E5 documented as unavailable due to missing eligible historical news.

Validation: full backend suite passed (62 tests); Python compilation passed; `git diff --check` passed; Vite production build passed (2,211 modules).

## Phase 9 - Final research experiments and evidence

**Status: completed with E5 unavailable by the documented data gate.** Generated `research/real_data/phase9_final_experiment_matrix.md` and `.json`, plus pooled walk-forward confusion matrix, reliability diagram, and cost-adjusted trading equity curve SVGs. The report consolidates all seven required experiments, methods, metrics, risks, and artifact links.

- **E1:** Five tickers across four sectors and four expanding folds. Equal-ticker mean accuracy/balanced accuracy/AUC/Brier: XGBoost 0.5333/0.5330/0.5179/0.2906; RF 0.5067/0.5114/0.5176/0.2607; LSTM 0.5236/0.4902/0.4806/0.2513; fixed hybrid 0.5165/0.4982/0.5092/0.2509; previous direction 0.5229/0.5117/0.5117/0.4771. XGBoost accuracy lead coincides with worse Brier. Final holdout results are not used in these fold summaries and remain exploratory because the same holdout had prior Phase 1 analysis.
- **E2:** On the common Phase 9 input, AAPL OOS LSTM-only returned 25.42%, fixed hybrid 25.1038%; fixed Sharpe was slightly higher (1.2878 vs 1.2518), while its drawdown was larger (-16.1417% vs -13.7985%).
- **E3:** On that same input, fixed hybrid returned 25.1038% / 1.2878 Sharpe / -16.1417% drawdown; adaptive hybrid returned 14.2802% / 0.8238 / -18.7356%.
- **E4:** Paired AAPL run uses the exact same retained CSV (`phase9_e4_aapl_prices.csv`, SHA-256 `94bf5ba4720da5dbbf597317ee44ac0f143528f7fb0f9c13eaf79f19418c33e3`) and OOS dates in both arms. Regime-specific weight selection was used on 220 audit rows; five rows fell back to overall validation. Without-regime control used overall validation on all 225 audit rows. One signal label differed (2025-08-05 BUY with regime vs HOLD without); the executed equity curves and trading metrics were identical: both returned 14.2802%, Sharpe 0.8238, max drawdown -18.7356%. Thus there was no measurable portfolio effect in this sample.
- **E5:** Not evaluable. No eligible point-in-time historical news archive was available. No current headlines were inserted into historical rows; no sentiment increment claim is made.
- **E6:** Raw fixed Brier 0.2532; calibrated fixed Brier 0.2676 (worse). No Trade abstained on 37/225 rows, returning 10.1851%, Sharpe 0.7945, drawdown -13.7985%; fixed returned 25.1038%, Sharpe 1.2878, drawdown -16.1417%. Drawdown decreased but return and Sharpe also decreased. Live threshold remains unavailable.
- **E7:** AAPL fixed hybrid returned 25.1038%, buy-and-hold 22.7918%, cash 0%; LSTM-only returned 25.4199%. Hybrid's drawdown exceeded buy-and-hold. These are cost-adjusted historical simulations, not evidence of future profitability.

The E4 runner now supports `--without-regime-weights`, `--input-csv`, and `--save-data`; it preserves paired data identity and reports regime context accurately. All AAPL strategy experiments E2–E7 use the common retained CSV with normalized frame SHA-256 `94bf5ba4720da5dbbf597317ee44ac0f143528f7fb0f9c13eaf79f19418c33e3`. `tools/build_phase9_report.py` validates the retained CSV hash, E4 data hashes, and date identity before producing the report and plots without extra plotting dependencies.

### Validation

- `tools/run_adaptive_evaluation.py` ran regime-enabled and disabled arms against the same saved CSV; both returned `status=ok`, four folds, and identical OOS dates and source hash.
- `tools/build_phase9_report.py` generated the final Markdown/JSON reports and SVG plots; its paired-data assertion passed.
- `\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v` - **62 tests passed**.
- `\.venv\Scripts\python.exe -m py_compile backend\hybrid_backtesting.py backend\tests\test_hybrid_backtesting.py tools\run_adaptive_evaluation.py tools\build_phase9_report.py` - passed.
- `\.venv\Scripts\python.exe tools\build_phase9_report.py` - passed; generated final Markdown/JSON and both SVG figures; shared E4 input assertion passed.
- `git diff --check` - passed. Frontend was not modified in Phase 9; Phase 8 production build remains passing.

### Acceptance checklist

- [x] E1–E7 covered; E5 explicitly not evaluable for the documented archive limitation.
- [x] Comparable chronological folds, target, and costs are reported where the source experiments support a paired comparison.
- [x] Per-ticker E1 fold summaries and aggregate metrics retained; AAPL experiments include return, Sharpe, drawdown, turnover/trades where available.
- [x] Negative and inconclusive findings retained.
- [x] Confusion matrices, reliability diagram, equity curves, trade logs, and audit reports are linked or embedded in machine-readable source artifacts.
- [x] E4 refuses to generate the final report if paired data hash or OOS dates differ.
- [x] Final holdout reuse caveat is retained; it is not used for Phase 9 walk-forward conclusions.
