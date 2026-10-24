import React, {
  createContext,
  lazy,
  Suspense,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import { createRoot } from "react-dom/client";
import {
  BrowserRouter,
  useNavigate,
  useLocation,
  Routes,
  Route,
  Link,
} from "react-router-dom";
import {
  Activity,
  BarChart3,
  Brain,
  FlaskConical,
  History,
  LayoutDashboard,
  Menu,
  Settings,
  Sliders,
  Sparkles,
  X,
  RefreshCw,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Sun,
  Moon,
} from "lucide-react";
import "./styles.css";

const SymbolContext = createContext(null);
const LineChartPanel = lazy(() =>
  import("./Charts.jsx").then((module) => ({ default: module.Chart })),
);
const FeatureChartPanel = lazy(() =>
  import("./Charts.jsx").then((module) => ({ default: module.FeatureChart })),
);

function useStockSymbol() {
  const context = useContext(SymbolContext);
  if (!context) throw new Error("Stock symbol context is missing.");
  return context;
}

const api = async (path) => {
  const r = await fetch(path);
  if (!r.ok) throw new Error(await responseMessage(r));
  return r.json();
};
const responseMessage = async (response) => {
  try {
    const body = await response.json();
    return body.detail || `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
};
const apiPost = async (path, body) => {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await responseMessage(r));
  return r.json();
};
const nav = [
  ["Dashboard", "/", "dashboard", LayoutDashboard],
  ["Analysis", "/analysis", "analysis", Sparkles],
  ["Backtest", "/backtest", "backtest", Activity],
  ["Models", "/models", "models", Brain],
  ["Tuning", "/tuning", "tuning", Sliders],
  ["Signals", "/signals", "signals", History],
  ["Research", "/research", "research", FlaskConical],
  ["Settings", "/settings", "settings", Settings],
];
const stockNames = {
  AAPL: "Apple Inc.",
  MSFT: "Microsoft",
  GOOGL: "Alphabet",
  GOOG: "Alphabet",
  AMZN: "Amazon",
  NVDA: "NVIDIA",
  TSLA: "Tesla",
  META: "Meta Platforms",
  JPM: "JPMorgan Chase",
  NFLX: "Netflix",
};
function Layout({ children }) {
  const [open, setOpen] = useState(false);
  const [theme, setTheme] = useState(
    () => localStorage.getItem("deepsignal_theme") || "dark",
  );
  const loc = useLocation();
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("deepsignal_theme", theme);
  }, [theme]);
  useEffect(() => setOpen(false), [loc.pathname]);
  return (
    <div className="app">
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>
      <header>
        <div className="brand">
          <button
            className="mobile"
            type="button"
            aria-label={open ? "Close navigation menu" : "Open navigation menu"}
            aria-expanded={open}
            aria-controls="mobile-navigation"
            onClick={() => setOpen(!open)}
          >
            {open ? <X /> : <Menu />}
          </button>
          <div className="logo">
            <Brain size={17} />
          </div>
          <div>
            <b>Stock Signal generator</b>
            <small>Hybrid DL Framework</small>
          </div>
        </div>
        <nav className="desktop" aria-label="Main navigation">
          {nav.map(([n, p, k, I]) => (
            <Link className={loc.pathname === p ? "active" : ""} key={k} to={p}>
              <I size={15} />
              {n}
            </Link>
          ))}
        </nav>
        <div className="header-actions">
          <button
            className="theme-toggle"
            type="button"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            <span>{theme === "dark" ? "Light" : "Dark"}</span>
          </button>
          <span
            className="live"
            title="Status refreshes periodically. Market prices use the latest available daily data."
          >
            <i /> Live
          </span>
        </div>
      </header>
      <nav
        className="mobileNav"
        id="mobile-navigation"
        aria-label="Main navigation"
        hidden={!open}
      >
        {nav.map(([n, p, k, I]) => (
          <Link onClick={() => setOpen(false)} key={k} to={p}>
            <I size={16} />
            {n}
          </Link>
        ))}
      </nav>
      <main id="main-content" tabIndex={-1}>
        {children}
      </main>
    </div>
  );
}
function Card({ children, className = "" }) {
  return <section className={"card " + className}>{children}</section>;
}
function Stat({ label, value, sub, icon: Icon }) {
  return (
    <Card>
      <div className="stat">
        <Icon />
        <div>
          <small>{label}</small>
          <strong>{value}</strong>
          <span>{sub}</span>
        </div>
      </div>
    </Card>
  );
}
function SignalBadge({ type }) {
  return (
    <span className={"badge " + type}>
      {type?.replace("_", " ").toUpperCase()}
    </span>
  );
}
function SymbolField({ value, onChange, compact = false, onSubmit, disabled }) {
  const inputId = useId();
  const listId = useId();
  const normalizedSymbol = value.trim().toUpperCase();
  const [resolvedName, setResolvedName] = useState(
    stockNames[normalizedSymbol] || "",
  );
  const savedSymbols = [
    ...new Set([
      "AAPL",
      ...loadSignals()
        .map((signal) => signal.stock_symbol?.trim().toUpperCase())
        .filter(Boolean),
    ]),
  ].sort((a, b) => (a === "AAPL" ? -1 : b === "AAPL" ? 1 : a.localeCompare(b)));

  useEffect(() => {
    let active = true;
    const knownName = stockNames[normalizedSymbol];
    if (!normalizedSymbol) {
      setResolvedName("");
      return () => {
        active = false;
      };
    }
    if (knownName) {
      setResolvedName(knownName);
      return () => {
        active = false;
      };
    }

    setResolvedName("");
    const timer = setTimeout(async () => {
      try {
        const result = await api(
          `/api/company/${encodeURIComponent(normalizedSymbol)}`,
        );
        if (active) {
          setResolvedName(result.company_name || "Company name unavailable");
        }
      } catch {
        if (active) setResolvedName("Company name unavailable");
      }
    }, 450);

    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [normalizedSymbol]);

  return (
    <div className={`symbol-field-group${compact ? " compact" : ""}`}>
      <label className="symbol-field" htmlFor={inputId}>
        <span>Stock symbol</span>
        <input
          id={inputId}
          value={value}
          disabled={disabled}
          list={listId}
          aria-describedby={`${inputId}-help`}
          onChange={(event) => onChange(event.target.value.toUpperCase())}
          onKeyDown={(event) => {
            if (event.key === "Enter" && onSubmit) onSubmit();
            if (event.key === " ") event.preventDefault();
          }}
          placeholder="Choose saved or type a ticker"
          aria-label="Stock ticker symbol"
          autoCapitalize="characters"
          autoComplete="off"
          maxLength={16}
          spellCheck={false}
        />
        <datalist id={listId}>
          {savedSymbols.map((ticker) => (
            <option
              key={ticker}
              value={ticker}
              label={stockNames[ticker] || "Saved signal"}
            />
          ))}
        </datalist>
        <small className="symbol-help" id={`${inputId}-help`}>
          Select a saved ticker or type a new one.
        </small>
      </label>
      {normalizedSymbol && (
        <output className="symbol-company" aria-live="polite">
          {resolvedName || "Looking up company…"}
        </output>
      )}
    </div>
  );
}
function Analysis() {
  const { symbol, setSymbol } = useStockSymbol();
  const [tf, setTf] = useState("1D");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  useEffect(() => {
    setData(null);
    setErr("");
  }, [symbol]);
  const run = async () => {
    setLoading(true);
    setErr("");
    setData(null);
    try {
      const r = await api(
        `/api/predict/${encodeURIComponent(symbol.trim())}?timeframe=${tf}`,
      );
      setData(r);
      saveSignal(r, tf);
    } catch (e) {
      setErr(
        `Could not analyze ${symbol.trim() || "this stock"}: ${e.message}`,
      );
    } finally {
      setLoading(false);
    }
  };
  return (
    <Page
      title="Stock Analysis"
      subtitle="Review the latest available data and an experimental next-session estimate."
    >
      <div className="toolbar">
        <SymbolField
          value={symbol}
          onChange={setSymbol}
          onSubmit={run}
          disabled={loading}
        />
        <select
          aria-label="Prediction timeframe"
          value={tf}
          onChange={(e) => setTf(e.target.value)}
        >
          <option>1D</option>
          <option>1W</option>
          <option>1M</option>
          <option>3M</option>
        </select>
        <button onClick={run} disabled={loading || !symbol.trim()}>
          {loading ? <RefreshCw className="spin" /> : <Sparkles />}
          {loading ? "Analyzing…" : "Analyze"}
        </button>
      </div>
      {err && <ErrorBox text={err} onRetry={run} retrying={loading} />}
      {loading && (
        <Card className="loader">
          <div className="spinner" />
          <b>Running hybrid deep learning analysis…</b>
          <span>
            LSTM → FinBERT → technical features → Random Forest → signal
          </span>
        </Card>
      )}
      {data && !loading && <Prediction data={data} timeframe={tf} />}
    </Page>
  );
}
function Prediction({ data, timeframe }) {
  const chart = (data.backtest?.dates || [])
    .map((d, i) => ({
      date: d,
      Strategy: data.backtest.strategy_curve[i],
      BuyHold: data.backtest.buy_hold[i],
    }))
    .slice(-120);
  return (
    <div className="stack">
      <Card>
        <div className="predictionHead">
          <div>
            <h2>{data.stock_symbol}</h2>
            <p>
              Latest available close: <b>{data.latest_close}</b> · data as of{" "}
              {data.as_of}
            </p>
          </div>
          <SignalBadge type={data.signal_type} />
        </div>
        <div className="stats4">
          <div>
            <small>Trend</small>
            <strong className={data.predicted_trend}>
              {data.predicted_trend}
            </strong>
          </div>
          <div>
            <small>Predicted change</small>
            <strong className={data.predicted_change >= 0 ? "up" : "down"}>
              {data.predicted_change >= 0 ? "+" : ""}
              {data.predicted_change}%
            </strong>
          </div>
          <div>
            <small>FinBERT sentiment</small>
            <strong>{data.sentiment_score}</strong>
          </div>
          <div>
            <small>Confidence</small>
            <strong>{data.confidence}%</strong>
          </div>
        </div>
        <div className="confidence">
          <span>Model confidence</span>
          <div
            role="progressbar"
            aria-label="Model confidence, not chance of profit"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={data.confidence}
          >
            <i style={{ width: `${data.confidence}%` }} />
          </div>
        </div>
      </Card>
      <Card className="prediction-explainer">
        <div>
          <h3>How to read this estimate</h3>
          <p>
            The model estimates whether the next session may close above or
            below the latest available close. The predicted change is an
            estimate, not a promised return.
          </p>
        </div>
        <div className="explanation-grid">
          <p>
            <b>Trend</b> is the model's up or down direction estimate.
          </p>
          <p>
            <b>Confidence</b> describes the model's signal strength. It is not
            the chance of making a profit.
          </p>
          <p>
            <b>Data as of</b> shows the latest price date used; market data can
            arrive late.
          </p>
        </div>
      </Card>
      {data.signal_explanation && (
        <SignalEvidence explanation={data.signal_explanation} />
      )}
      <div className="two">
        <Card>
          <h3>Latest Backtest</h3>
          <Suspense fallback={<ChartLoading />}>
            <LineChartPanel data={chart} />
          </Suspense>
        </Card>
        <Card>
          <h3>Technical Indicators</h3>
          <div className="indicatorGrid">
            {Object.entries(data.technical_indicators).map(([k, v]) => (
              <div key={k}>
                <small>{k.replaceAll("_", " ")}</small>
                <b>{v}</b>
              </div>
            ))}
          </div>
        </Card>
      </div>
      <div className="two">
        <Card>
          <h3>FinBERT News Sentiment</h3>
          {data.news?.length ? (
            <div className="news">
              {data.news.map((n, i) => (
                <div key={i}>
                  <b>{n.title}</b>
                  <small>
                    {n.source || "News"} ·{" "}
                    {n.sentiment == null
                      ? "FinBERT unavailable"
                      : `score ${n.sentiment}`}
                  </small>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">No recent news was returned.</p>
          )}
        </Card>
        <Card>
          <h3>Analysis & Risks</h3>
          <p>{data.analysis_summary}</p>
          <ul>
            {data.risk_factors?.map((x, i) => (
              <li key={i}>{x}</li>
            ))}
          </ul>
          <p className="muted">{data.model_recommendation}</p>
        </Card>
      </div>
      <Card>
        <h3>Feature Importance</h3>
        <Suspense fallback={<ChartLoading />}>
          <FeatureChartPanel
            data={Object.entries(data.feature_importance || {})
              .map(([feature, value]) => ({ feature, value }))
              .sort((a, b) => b.value - a.value)}
          />
        </Suspense>
      </Card>
      <p className="disclaimer">
        This is an experimental prediction system. It is not financial advice
        and does not guarantee future performance.
      </p>
    </div>
  );
}
function SignalEvidence({ explanation }) {
  const labels = {
    lstm: "LSTM",
    random_forest: "Random Forest",
    finbert: "FinBERT sentiment",
    technical: "Technical indicators",
  };
  const order = ["lstm", "random_forest", "finbert", "technical"];
  return (
    <Card className="signal-evidence">
      <div className="evidence-heading">
        <div>
          <h3>Signal Evidence</h3>
          <p>Weighted model inputs behind the current recommendation.</p>
        </div>
        <strong>
          {(explanation.ensemble_probability_up * 100).toFixed(1)}% combined
          bullish score
        </strong>
      </div>
      <div className="evidence-grid">
        {order.map((key) => {
          const part = explanation.components?.[key];
          if (!part) return null;
          const probability = Math.max(
            0,
            Math.min(1, Number(part.probability_up) || 0),
          );
          const contribution = Math.max(
            0,
            Number(part.weighted_contribution) || 0,
          );
          return (
            <div className="evidence-item" key={key}>
              <div className="evidence-item-head">
                <b>{labels[key]}</b>
                <span>
                  {(probability * 100).toFixed(1)}% up ·{" "}
                  {(Number(part.weight) * 100).toFixed(0)}% weight
                </span>
              </div>
              <div className="evidence-track">
                <i style={{ width: `${probability * 100}%` }} />
              </div>
              <small>
                Weighted contribution: {(contribution * 100).toFixed(1)} points
              </small>
              {key === "finbert" && (
                <small>
                  Raw sentiment: {Number(part.sentiment_score).toFixed(3)} (−1
                  bearish to +1 bullish)
                </small>
              )}
              {key === "technical" && (
                <ul>
                  {(part.factors || []).map((factor, i) => (
                    <li key={i}>{factor}</li>
                  ))}
                </ul>
              )}
            </div>
          );
        })}
      </div>
      <p className="evidence-note">
        The score combines model probabilities and current indicators; it is an
        estimate, not a measured chance of profit.
      </p>
    </Card>
  );
}
function ChartLoading() {
  return (
    <div className="chart-loading" role="status" aria-live="polite">
      Loading chart...
    </div>
  );
}
function Dashboard() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [market, setMarket] = useState([]);
  const [marketError, setMarketError] = useState(false);
  const [marketLoading, setMarketLoading] = useState(false);
  const [marketCheckedAt, setMarketCheckedAt] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobStatusUnavailable, setJobStatusUnavailable] = useState(false);
  const [jobStatusRetry, setJobStatusRetry] = useState(0);
  const [marketRetry, setMarketRetry] = useState(0);
  const { symbol, setSymbol } = useStockSymbol();
  const lastLoadedSymbol = useRef(symbol);
  useEffect(() => {
    const saved = loadSignals();
    setItems(saved);
    setLoading(false);
  }, []);
  useEffect(() => {
    if (lastLoadedSymbol.current !== symbol) {
      setMarket([]);
      setMarketCheckedAt(null);
      setMarketError(false);
      lastLoadedSymbol.current = symbol;
    }
  }, [symbol]);
  useEffect(() => {
    let active = true;
    const refresh = async () => {
      if (!symbol.trim()) {
        setMarket([]);
        setMarketError(true);
        setMarketLoading(false);
        return;
      }
      setMarketLoading(true);
      try {
        const r = await api(
          `/api/history/${encodeURIComponent(symbol.trim())}?period=3mo`,
        );
        if (active) {
          setMarket(r.data || []);
          setMarketError(false);
          setMarketCheckedAt(new Date());
        }
      } catch {
        if (active) setMarketError(true);
      } finally {
        if (active) setMarketLoading(false);
      }
    };
    const debounce = setTimeout(refresh, 450);
    const timer = setInterval(refresh, 15 * 60 * 1000);
    return () => {
      active = false;
      clearTimeout(debounce);
      clearInterval(timer);
    };
  }, [symbol, marketRetry]);
  useEffect(() => {
    let active = true;
    const refreshStatus = async () => {
      try {
        const r = await api("/api/jobs/market-data");
        if (active) {
          setJobStatus(r);
          setJobStatusUnavailable(false);
        }
      } catch {
        if (active) setJobStatusUnavailable(true);
      }
    };
    refreshStatus();
    const timer = setInterval(refreshStatus, 60 * 1000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [jobStatusRetry]);
  const buy = items.filter((x) => x.signal_type?.includes("buy")).length;
  const sell = items.filter((x) => x.signal_type?.includes("sell")).length;
  const avg = items.length
    ? items.reduce((a, x) => a + (x.confidence || 0), 0) / items.length
    : 0;
  const change =
    market.length > 1 ? (market.at(-1).close / market[0].close - 1) * 100 : 0;
  const sentiment = Math.max(-1, Math.min(1, change / 8));
  const company = stockNames[symbol] || symbol;
  const latestSavedSignal = items.find(
    (item) => item.stock_symbol?.toUpperCase() === symbol.toUpperCase(),
  );
  const savedSignalType = latestSavedSignal?.signal_type?.toLowerCase() || "";
  const savedSignalAction = savedSignalType.includes("buy")
    ? "BUY"
    : savedSignalType.includes("sell")
      ? "SELL"
      : savedSignalType.includes("hold")
        ? "HOLD"
        : null;
  return (
    <Page
      title="Dashboard"
      subtitle="Daily market data with periodic status updates."
    >
      <div className="stats4">
        <Stat
          label="Saved Signals"
          value={items.length}
          sub="Browser history"
          icon={History}
        />
        <Stat
          label="Buy Signals"
          value={buy}
          sub={`${sell} sell signals`}
          icon={TrendingUp}
        />
        <Stat
          label="Avg Confidence"
          value={`${avg.toFixed(1)}%`}
          sub="Saved predictions"
          icon={Brain}
        />
        <Stat
          label="Data Source"
          value="Yahoo"
          sub="Latest available bars"
          icon={Activity}
        />
      </div>
      <div className="dashboard-grid">
        <Card className="market-card">
          <div className="market-title">
            <div>
              <div className="eyebrow">MARKET OVERVIEW · 3 MONTHS</div>
              <div className="market-stock-heading">
                <SymbolField
                  value={symbol}
                  onChange={setSymbol}
                  compact
                  onSubmit={() => setMarketRetry((value) => value + 1)}
                />
              </div>
            </div>
            <div className="market-price">
              {market.length ? `$${market.at(-1).close.toFixed(2)}` : "—"}
              <small className={change >= 0 ? "up" : "down"}>
                {market.length
                  ? `${change >= 0 ? "+" : ""}${change.toFixed(2)}% in period`
                  : "Waiting for market data"}
              </small>
            </div>
          </div>
          <div
            className="market-update"
            role="group"
            aria-label="Market data update status"
          >
            <span>
              {market.length
                ? `Daily bars through ${market.at(-1).date}`
                : "No market date available"}
            </span>
            <span>
              {marketCheckedAt
                ? `Chart checked ${marketCheckedAt.toLocaleTimeString()}`
                : "Checking chart\u2026"}
            </span>
            <span>
              {jobStatusUnavailable
                ? "Background refresh status unavailable"
                : jobStatus?.last_run
                  ? `Background refresh: ${jobStatus.last_run.status} \u00b7 ${new Date(jobStatus.last_run.completed_at || jobStatus.last_run.started_at).toLocaleString()}`
                  : "Background refresh has not run yet"}
            </span>
            {jobStatusUnavailable && (
              <button
                className="text-button"
                type="button"
                onClick={() => setJobStatusRetry((value) => value + 1)}
              >
                Retry status check
              </button>
            )}
            <span>
              Yahoo daily data may be delayed or unavailable on market holidays.
            </span>
          </div>
          {marketError && (
            <div className="market-warning" role="alert">
              <span>
                {market.length
                  ? "Could not update the chart; the last loaded data is still shown."
                  : "Could not load market data. Check the ticker and backend connection."}
              </span>
              <button
                className="text-button"
                type="button"
                onClick={() => setMarketRetry((value) => value + 1)}
              >
                Try again
              </button>
            </div>
          )}
          {market.length ? (
            <CandleChart data={market} symbol={symbol} />
          ) : (
            <div className="market-empty">
              {marketError
                ? "Market history unavailable. Check that the backend is running."
                : marketLoading
                  ? "Loading market history…"
                  : "No market history available."}
            </div>
          )}
          <div className="market-indicators">
            <div>
              <small>RSI · 14</small>
              <b>
                {market.length
                  ? indicatorRsi(market.map((x) => x.close)).toFixed(1)
                  : "—"}
              </b>
              <span>Momentum</span>
            </div>
            <div>
              <small>MACD · 12/26</small>
              <b>
                {market.length
                  ? macd(market.map((x) => x.close)).toFixed(2)
                  : "—"}
              </b>
              <span>Trend strength</span>
            </div>
            <div>
              <small>AVG VOLUME</small>
              <b>
                {market.length
                  ? (
                      market.slice(-20).reduce((a, x) => a + x.volume, 0) /
                      Math.min(20, market.length) /
                      1e6
                    ).toFixed(1) + "M"
                  : "—"}
              </b>
              <span>Last 20 sessions</span>
            </div>
          </div>
        </Card>
        <Card className="sentiment-card">
          <div className="eyebrow">MARKET SENTIMENT</div>
          <h3>Price momentum</h3>
          <SentimentGauge score={sentiment} />
          <p>Directional reading based on {company}’s 3-month price change.</p>
          <div className="sentiment-legend">
            <span>Bearish</span>
            <span>Neutral</span>
            <span>Bullish</span>
          </div>
          <div className="sentiment-value">
            {sentiment > 0.18
              ? "🙂 Bullish"
              : sentiment < -0.18
                ? "🙁 Bearish"
                : "😐 Neutral"}{" "}
            <small>({sentiment.toFixed(2)})</small>
          </div>
          <div className="dashboard-saved-signal">
            <div className="dashboard-saved-signal-heading">
              <div>
                <div className="eyebrow">LATEST SAVED SIGNAL</div>
                <strong>{symbol} model output</strong>
              </div>
              {savedSignalAction && (
                <span
                  className={`dashboard-signal-badge ${savedSignalAction.toLowerCase()}`}
                >
                  {savedSignalAction}
                </span>
              )}
            </div>
            {loading ? (
              <p>Loading saved signals...</p>
            ) : latestSavedSignal && savedSignalAction ? (
              <>
                <p>
                  Saved analysis result for {symbol}. It is not a guarantee of
                  future performance.
                </p>
                <div className="dashboard-saved-signal-meta">
                  <span>
                    {latestSavedSignal.confidence != null &&
                    Number.isFinite(Number(latestSavedSignal.confidence))
                      ? `${Number(latestSavedSignal.confidence).toFixed(1)}% model confidence`
                      : "Model confidence unavailable"}
                  </span>
                  <span>
                    Saved {latestSavedSignal.saved_at || "date unavailable"}
                  </span>
                </div>
              </>
            ) : (
              <>
                <p>No saved BUY, HOLD, or SELL signal for {symbol} yet.</p>
                <Link to="/analysis" className="button secondary">
                  Generate a signal
                </Link>
              </>
            )}
          </div>
        </Card>
      </div>
      <Card>
        <h3>System Architecture</h3>
        <div className="pipeline">
          {[
            "yfinance / News API",
            "Technical Indicators",
            "LSTM Forecast",
            "FinBERT Sentiment",
            "Random Forest",
            "BUY / SELL / HOLD",
            "Walk-Forward Backtest",
          ].map((x, i) => (
            <div key={x}>
              {i + 1}
              <b>{x}</b>
            </div>
          ))}
        </div>
      </Card>
      <Card>
        <div className="row">
          <h3>Latest Trading Signals</h3>
          <Link to="/analysis" className="button secondary">
            New analysis
          </Link>
        </div>
        {loading ? (
          <p>Loading…</p>
        ) : items.length ? (
          <div className="signalGrid">
            {items.slice(0, 6).map((x, i) => (
              <SignalMini key={i} x={x} />
            ))}
          </div>
        ) : (
          <Empty />
        )}
      </Card>
    </Page>
  );
}
function CandleChart({ data, symbol }) {
  const rows = data.slice(-60),
    w = 760,
    h = 250,
    p = { l: 48, r: 16, t: 12, b: 25 };
  const vals = rows.flatMap((x) => [x.high, x.low]),
    min = Math.min(...vals),
    max = Math.max(...vals),
    range = max - min || 1;
  const y = (v) => p.t + ((max - v) / range) * (h - p.t - p.b),
    step = (w - p.l - p.r) / rows.length,
    bw = Math.max(3, step * 0.55);
  return (
    <div className="candle-wrap">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label={`${symbol} daily candlestick price chart`}
      >
        <defs>
          <linearGradient id="volumeFade" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#8b9cfb" stopOpacity=".25" />
            <stop offset="1" stopColor="#8b9cfb" stopOpacity=".03" />
          </linearGradient>
        </defs>
        {Array.from({ length: 5 }, (_, i) => {
          const yy = p.t + (i * (h - p.t - p.b)) / 4,
            v = max - (i * range) / 4;
          return (
            <g key={i}>
              <line
                x1={p.l}
                x2={w - p.r}
                y1={yy}
                y2={yy}
                stroke="#e9edf3"
                strokeDasharray="3 4"
              />
              <text x={p.l - 8} y={yy + 4} textAnchor="end">
                {v.toFixed(0)}
              </text>
            </g>
          );
        })}
        {rows.map((d, i) => {
          const x = p.l + step * i + step / 2,
            up = d.close >= d.open,
            top = y(Math.max(d.open, d.close)),
            bot = y(Math.min(d.open, d.close)),
            vh = Math.min(
              35,
              (d.volume / (Math.max(...rows.map((a) => a.volume)) || 1)) * 35,
            );
          return (
            <g key={d.date}>
              <rect
                x={x - bw / 2}
                y={h - p.b - vh}
                width={bw}
                height={vh}
                fill="url(#volumeFade)"
              />
              <line
                x1={x}
                x2={x}
                y1={y(d.high)}
                y2={y(d.low)}
                stroke={up ? "#10b981" : "#ef4444"}
                strokeWidth="1.2"
              />
              <rect
                x={x - bw / 2}
                y={top}
                width={bw}
                height={Math.max(1.5, bot - top)}
                fill={up ? "#10b981" : "#ef4444"}
                rx=".5"
              />
            </g>
          );
        })}
        <text x={p.l} y={h - 4}>
          {rows[0]?.date?.slice(5)}
        </text>
        <text x={w - p.r} y={h - 4} textAnchor="end">
          {rows.at(-1)?.date?.slice(5)}
        </text>
      </svg>
      <div className="chart-legend">
        <span>
          <i className="green" /> Up
        </span>
        <span>
          <i className="red" /> Down
        </span>
        <span>
          <i className="volume-key" /> Volume
        </span>
        <em>Daily OHLC · USD</em>
      </div>
    </div>
  );
}
function SentimentGauge({ score }) {
  const angle = -90 + score * 90,
    rad = (angle * Math.PI) / 180,
    x = 150 + 110 * Math.cos(rad),
    y = 135 + 110 * Math.sin(rad);
  return (
    <svg
      className="gauge"
      viewBox="0 0 300 170"
      role="img"
      aria-label={`Momentum sentiment score ${score.toFixed(2)}`}
    >
      <path
        d="M40 135 A110 110 0 0 1 260 135"
        fill="none"
        stroke="#edf0f5"
        strokeWidth="28"
      />
      <path
        d="M40 135 A110 110 0 0 1 150 25"
        fill="none"
        stroke="#ef4444"
        strokeWidth="28"
      />
      <path
        d="M150 25 A110 110 0 0 1 260 135"
        fill="none"
        stroke="#10b981"
        strokeWidth="28"
      />
      <line
        x1="150"
        y1="135"
        x2={x}
        y2={y}
        stroke="#374151"
        strokeWidth="5"
        strokeLinecap="round"
      />
      <circle cx="150" cy="135" r="8" fill="#374151" />
      <text x="32" y="158">
        −1
      </text>
      <text x="150" y="15" textAnchor="middle">
        0
      </text>
      <text x="268" y="158" textAnchor="end">
        +1
      </text>
    </svg>
  );
}
function indicatorRsi(values) {
  const tail = values.slice(-15),
    diffs = tail.slice(1).map((v, i) => v - tail[i]),
    up = diffs.filter((x) => x > 0).reduce((a, x) => a + x, 0) / 14,
    down = Math.abs(diffs.filter((x) => x < 0).reduce((a, x) => a + x, 0)) / 14;
  return down === 0 ? 100 : 100 - 100 / (1 + up / down);
}
function macd(values) {
  const ema = (period) => {
    const a = 2 / (period + 1);
    return values.reduce((v, x, i) => (i ? x * a + v * (1 - a) : x), 0);
  };
  return ema(12) - ema(26);
}
function SignalMini({ x }) {
  return (
    <div className="signal">
      <div className="row">
        <b>{x.stock_symbol}</b>
        <SignalBadge type={x.signal_type} />
      </div>
      <span>
        {x.predicted_trend} · {x.confidence}% confidence
      </span>
      <small>{x.saved_at}</small>
    </div>
  );
}
function Backtest() {
  const { symbol, setSymbol } = useStockSymbol();
  const [d, setD] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    setD(null);
    setErr("");
  }, [symbol]);
  const run = async () => {
    setLoading(true);
    setErr("");
    setD(null);
    try {
      setD(await api(`/api/backtest/${encodeURIComponent(symbol.trim())}`));
    } catch (e) {
      setErr(
        `Could not backtest ${symbol.trim() || "this stock"}: ${e.message}`,
      );
    } finally {
      setLoading(false);
    }
  };
  return (
    <Page
      title="Historical Backtesting"
      subtitle="Past-only signals execute at the next session open; costs, slippage and assumptions are included."
    >
      <div className="toolbar">
        <SymbolField
          value={symbol}
          onChange={setSymbol}
          onSubmit={run}
          disabled={loading}
        />
        <button onClick={run} disabled={loading || !symbol.trim()}>
          {loading ? <RefreshCw className="spin" /> : <Activity />}
          {loading ? "Running backtest..." : "Run backtest"}
        </button>
      </div>
      {err && <ErrorBox text={err} onRetry={run} retrying={loading} />}
      {d && (
        <div className="stack">
          <Card>
            <p className="muted">
              {d.strategy} · {d.assumptions.execution} ·{" "}
              {d.assumptions.transaction_cost_pct}% fee ·{" "}
              {d.assumptions.slippage_pct}% slippage
            </p>
            <Suspense fallback={<ChartLoading />}>
              <LineChartPanel
                data={d.dates
                  .map((date, i) => ({
                    date,
                    Strategy: d.strategy_curve[i],
                    BuyHold: d.buy_hold[i],
                    NoTrade: d.no_trade[i],
                  }))
                  .slice(-180)}
              />
            </Suspense>
          </Card>
          <div className="three">
            {[
              ["Strategy", d.summary.strategy],
              ["Buy & Hold", d.summary.buy_hold],
              ["No Trade", d.summary.no_trade],
            ].map(([n, m]) => (
              <Card key={n}>
                <h3>{n}</h3>
                <div className="metric">
                  <b>{m.total_return_pct}%</b>
                  <span>Return</span>
                </div>
                <div className="metric">
                  <b>{m.annualized_sharpe}</b>
                  <span>Sharpe</span>
                </div>
                <div className="metric">
                  <b>{m.max_drawdown_pct}%</b>
                  <span>Max drawdown</span>
                </div>
                <div className="metric">
                  <b>{m.trade_count}</b>
                  <span>Executed trades</span>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}
    </Page>
  );
}
function Research() {
  return (
    <Page
      title="Research & Methodology"
      subtitle="Project proposal phases implemented in the application."
    >
      <div className="research">
        {[
          [
            "1",
            "Data Collection",
            "yfinance for OHLCV; News API when configured; Yahoo Finance fallback.",
          ],
          [
            "2",
            "Feature Engineering",
            "SMA, RSI, MACD, Bollinger Bands and time-aligned FinBERT sentiment.",
          ],
          [
            "3",
            "Price Forecasting",
            "TensorFlow/Keras LSTM predicts short-term trend direction and is compared with naive behavior.",
          ],
          [
            "4",
            "Signal Classification",
            "Random Forest fuses LSTM probability, sentiment and technical features; XGBoost is available for comparison.",
          ],
          [
            "5",
            "Backtesting",
            "Walk-forward evaluation with transaction costs/slippage against buy-and-hold and naive baselines.",
          ],
          [
            "6",
            "Explainability",
            "Feature importance is shown in the UI; SHAP can be added to the model service for deeper local explanations.",
          ],
        ].map(([n, t, p]) => (
          <Card key={n}>
            <span className="phase">PHASE {n}</span>
            <h2>{t}</h2>
            <p>{p}</p>
          </Card>
        ))}
      </div>
    </Page>
  );
}
function Models() {
  const { symbol, setSymbol } = useStockSymbol();
  const [d, setD] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  useEffect(() => {
    setD(null);
    setErr("");
  }, [symbol]);
  const run = async () => {
    setLoading(true);
    setErr("");
    setD(null);
    try {
      setD(
        await api(
          `/api/predict/${encodeURIComponent(symbol.trim())}?timeframe=1D`,
        ),
      );
    } catch (e) {
      setErr(`Could not load model comparison: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };
  const evaluation = d?.model_comparison?.evaluation;
  return (
    <Page
      title="Models"
      subtitle="Hybrid model components, comparison and explainability."
    >
      <div className="toolbar">
        <SymbolField
          value={symbol}
          onChange={setSymbol}
          onSubmit={run}
          disabled={loading}
        />
        <button onClick={run} disabled={loading || !symbol.trim()}>
          {loading ? "Loading..." : "Compare models"}
        </button>
      </div>
      {err && <ErrorBox text={err} onRetry={run} retrying={loading} />}
      <div className="three">
        {[
          ["LSTM", "Time-series trend probability", "TensorFlow / Keras"],
          [
            "FinBERT",
            "Finance-aware news sentiment",
            "Hugging Face Transformers",
          ],
          ["Random Forest", "Buy / sell / hold classifier", "Scikit-learn"],
        ].map(([a, b, c]) => (
          <Card key={a}>
            <Brain />
            <h2>{a}</h2>
            <p>{b}</p>
            <small>{c}</small>
          </Card>
        ))}
      </div>
      {d && (
        <div className="two">
          <Card>
            <h3>Model Comparison</h3>
            {evaluation?.status === "ok" ? (
              <p className="muted">
                Chronological holdout: trained on {evaluation.train_samples}{" "}
                earlier rows; evaluated on {evaluation.test_samples} later rows
                ({evaluation.test_start} to {evaluation.test_end}).
              </p>
            ) : (
              evaluation && (
                <p className="muted">
                  Not enough labeled history for a reliable holdout evaluation (
                  {evaluation.train_samples} samples available).
                </p>
              )
            )}
            {Object.entries(d.model_comparison || {})
              .filter(([k]) => !["shap", "evaluation"].includes(k))
              .map(([k, v]) => (
                <div key={k}>
                  <div className="metric">
                    <b>{v.accuracy ?? "—"}%</b>
                    <span>{k} holdout accuracy</span>
                  </div>
                  {v.balanced_accuracy != null && (
                    <div className="metric">
                      <b>{v.balanced_accuracy}%</b>
                      <span>{k} balanced accuracy</span>
                    </div>
                  )}
                </div>
              ))}
            <div className="selectedModelNote">
              <strong>Model used by the hybrid: Random Forest</strong>
            </div>
          </Card>
          <Card>
            <h3>SHAP Local Explanation</h3>
            {Object.entries(d.model_comparison?.shap || {})
              .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
              .slice(0, 8)
              .map(([k, v]) => (
                <div className="metric" key={k}>
                  <b>
                    {v > 0 ? "+" : ""}
                    {v}
                  </b>
                  <span>{k}</span>
                </div>
              ))}
          </Card>
        </div>
      )}
    </Page>
  );
}
function Tuning() {
  const defaults = {
    lookback: 30,
    lstm_units: 48,
    epochs: 6,
    rf_trees: 250,
    transaction_cost_pct: 0.06,
    slippage_pct: 0.04,
  };
  const [config, setConfig] = useState(defaults);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [loadingConfig, setLoadingConfig] = useState(false);
  const loadConfig = async () => {
    setLoadingConfig(true);
    setMessage("");
    try {
      setConfig(await api("/api/config"));
    } catch (e) {
      setMessage(`Could not load backend settings: ${e.message}`);
    } finally {
      setLoadingConfig(false);
    }
  };
  useEffect(() => {
    loadConfig();
  }, []);
  const update = (key, value) => {
    setConfig((current) => ({ ...current, [key]: value }));
    setMessage("");
  };
  const save = async () => {
    setSaving(true);
    setMessage("");
    try {
      const result = await apiPost(
        "/api/config",
        Object.fromEntries(
          Object.entries(config).map(([k, v]) => [k, Number(v)]),
        ),
      );
      setConfig(result.config);
      setMessage(
        "Experiment settings saved. New analyses and backtests will use them.",
      );
    } catch (e) {
      setMessage(`Could not save settings: ${e.message}`);
    } finally {
      setSaving(false);
    }
  };
  return (
    <Page
      title="Model Tuning"
      subtitle="Configure the model and backtest parameters used by the backend."
    >
      <Card>
        <div className="formGrid">
          <label>
            Lookback (10–120 days)
            <input
              value={config.lookback}
              onChange={(e) => update("lookback", e.target.value)}
              type="number"
              min="10"
              max="120"
            />
          </label>
          <label>
            LSTM units (8–256)
            <input
              value={config.lstm_units}
              onChange={(e) => update("lstm_units", e.target.value)}
              type="number"
              min="8"
              max="256"
            />
          </label>
          <label>
            Epochs (1–100)
            <input
              value={config.epochs}
              onChange={(e) => update("epochs", e.target.value)}
              type="number"
              min="1"
              max="100"
            />
          </label>
          <label>
            Random Forest trees (50–1000)
            <input
              value={config.rf_trees}
              onChange={(e) => update("rf_trees", e.target.value)}
              type="number"
              min="50"
              max="1000"
            />
          </label>
          <label>
            Transaction cost (%)
            <input
              value={config.transaction_cost_pct}
              onChange={(e) => update("transaction_cost_pct", e.target.value)}
              type="number"
              min="0"
              max="2"
              step="0.01"
            />
          </label>
          <label>
            Slippage (%)
            <input
              value={config.slippage_pct}
              onChange={(e) => update("slippage_pct", e.target.value)}
              type="number"
              min="0"
              max="2"
              step="0.01"
            />
          </label>
        </div>
        <button onClick={save} disabled={saving}>
          {saving ? "Saving…" : "Save experiment settings"}
        </button>
        {message.startsWith("Could not") ? (
          <ErrorBox
            text={message}
            onRetry={message.startsWith("Could not load") ? loadConfig : save}
            retrying={saving || loadingConfig}
          />
        ) : message ? (
          <p className="tuning-message" role="status">
            {message}
          </p>
        ) : null}
      </Card>
    </Page>
  );
}
function Signals() {
  const [items, setItems] = useState(loadSignals());
  return (
    <Page
      title="Signal History"
      subtitle="Locally saved predictions from the Analysis page."
    >
      <div className="row">
        <span className="muted">{items.length} saved signals</span>
        <button
          className="secondary"
          onClick={() => {
            localStorage.removeItem("deepsignal_signals");
            setItems([]);
          }}
        >
          Clear history
        </button>
      </div>
      {items.length ? (
        <div className="signalGrid">
          {items.map((x, i) => (
            <SignalMini key={i} x={x} />
          ))}
        </div>
      ) : (
        <Empty />
      )}
    </Page>
  );
}
function SettingsPage() {
  return (
    <Page title="Settings" subtitle="Local application settings.">
      <Card>
        <label>
          Backend URL
          <input value="/api" readOnly />
        </label>
        <p className="muted">
          The frontend uses the Vite proxy in development. Configure
          CORS_ORIGINS in the backend .env when deploying.
        </p>
      </Card>
    </Page>
  );
}
function Page({ title, subtitle, children }) {
  return (
    <div>
      <div className="pageHead">
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      {children}
    </div>
  );
}
function Empty() {
  return (
    <Card className="empty">
      <AlertTriangle />
      <p>No saved signals yet. Run an analysis first.</p>
      <Link to="/analysis" className="button">
        Analyze a stock
      </Link>
    </Card>
  );
}
function ErrorBox({ text, onRetry, retrying = false }) {
  return (
    <div className="error" role="alert">
      <AlertTriangle size={18} aria-hidden="true" />
      <span>{text}</span>
      {onRetry && (
        <button
          className="error-retry"
          type="button"
          onClick={onRetry}
          disabled={retrying}
        >
          {retrying ? "Retrying..." : "Try again"}
        </button>
      )}
    </div>
  );
}
function saveSignal(x, tf) {
  const arr = loadSignals();
  const item = { ...x, timeframe: tf, saved_at: new Date().toLocaleString() };
  localStorage.setItem(
    "deepsignal_signals",
    JSON.stringify(
      [
        item,
        ...arr.filter(
          (a) => !(a.stock_symbol === x.stock_symbol && a.timeframe === tf),
        ),
      ].slice(0, 100),
    ),
  );
}
function loadSignals() {
  try {
    return JSON.parse(localStorage.getItem("deepsignal_signals") || "[]");
  } catch {
    return [];
  }
}
function App() {
  const [symbol, setSymbol] = useState(
    () => localStorage.getItem("deepsignal_selected_symbol") || "AAPL",
  );

  useEffect(() => {
    localStorage.setItem("deepsignal_selected_symbol", symbol);
  }, [symbol]);

  return (
    <SymbolContext.Provider value={{ symbol, setSymbol }}>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/backtest" element={<Backtest />} />
            <Route path="/models" element={<Models />} />
            <Route path="/tuning" element={<Tuning />} />
            <Route path="/signals" element={<Signals />} />
            <Route path="/research" element={<Research />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<Dashboard />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </SymbolContext.Provider>
  );
}
createRoot(document.getElementById("root")).render(<App />);
