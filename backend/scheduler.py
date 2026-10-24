"""Small in-process scheduler for periodic market-data refreshes."""

import logging
import math
import os
import threading
import time
from datetime import datetime, timezone

try:
    from . import database as storage
except ImportError:  # Support ``uvicorn main:app`` from the backend directory.
    import database as storage

logger = logging.getLogger("deepsignal.scheduler")
_stop_event = threading.Event()
_thread = None
_state_lock = threading.Lock()
_state = {"running": False, "last_run": None}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _symbols():
    values = os.getenv("MARKET_DATA_SYMBOLS", "AAPL").strip() or "AAPL"
    symbols = list(dict.fromkeys(item.strip().upper().replace(" ", "") for item in values.split(",") if item.strip()))
    return symbols or ["AAPL"]


def refresh_market_data_once():
    """Fetch, validate, feature-engineer and persist one refresh cycle."""
    try:
        from .main import download_history, indicators
    except ImportError:
        from main import download_history, indicators

    started_at = _utc_now()
    logger.info("Starting market-data refresh for %s", ", ".join(_symbols()))
    total_rows = 0
    failures = {}
    for symbol in _symbols():
        for attempt in range(3):
            try:
                # A scheduled update must reach Yahoo even when an HTTP request
                # populated the short-lived request cache moments earlier.
                frame = download_history(symbol, "1y", force_refresh=True)
                if frame.empty or frame.isna().any().any():
                    raise ValueError("Provider returned an empty dataset or missing OHLCV values.")
                if frame["date"].duplicated().any():
                    raise ValueError("Provider returned duplicate daily timestamps.")
                if not frame["date"].is_monotonic_increasing:
                    raise ValueError("Provider returned out-of-order daily timestamps.")

                featured = indicators(frame)
                records = []
                for row in featured.to_dict(orient="records"):
                    features = {
                        key: float(value)
                        for key, value in row.items()
                        if key not in {"date", "open", "high", "low", "close", "volume"}
                        and value is not None and math.isfinite(float(value))
                    }
                    records.append({
                        "timestamp": row["date"].to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                        "features": features,
                        "source": "Yahoo Finance",
                        "retrieved_at": _utc_now(),
                    })
                saved = storage.save_market_bars(
                    symbol, "1d", records,
                    os.path.join(os.path.dirname(__file__), "market_data_bars.json"),
                )
                total_rows += saved
                logger.info("Market refresh saved %d daily bars for %s", saved, symbol)
                break
            except Exception as exc:
                if attempt == 2:
                    failures[symbol] = f"{type(exc).__name__}: {exc}"
                    logger.exception("Market refresh failed for %s after 3 attempts", symbol)
                else:
                    time.sleep(2 ** attempt)

    completed_at = _utc_now()
    result = {
        "job_name": "market_data_refresh",
        "started_at": started_at,
        "completed_at": completed_at,
        "status": "failed" if len(failures) == len(_symbols()) else "partial" if failures else "succeeded",
        "rows_processed": total_rows,
        "details": {"symbols": _symbols(), "failures": failures, "interval": "1d", "provider": "Yahoo Finance"},
    }
    try:
        storage.record_job_run(result, os.path.join(os.path.dirname(__file__), "job_runs.json"))
    except Exception:
        logger.exception("Could not persist the market refresh job status")
    with _state_lock:
        _state["last_run"] = result
    logger.info(
        "Market-data refresh %s: processed %d bars; %d ticker(s) failed",
        result["status"], total_rows, len(failures),
    )
    return result


def _run_forever():
    with _state_lock:
        _state["running"] = True
    try:
        while not _stop_event.is_set():
            refresh_market_data_once()
            try:
                minutes = max(5, int(os.getenv("MARKET_DATA_REFRESH_MINUTES", "360")))
            except ValueError:
                minutes = 360
                logger.warning("Invalid MARKET_DATA_REFRESH_MINUTES; using 360 minutes")
            if _stop_event.wait(minutes * 60):
                break
    finally:
        with _state_lock:
            _state["running"] = False


def start_scheduler():
    global _thread
    if os.getenv("MARKET_DATA_REFRESH_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
        logger.info("Scheduled market refresh is disabled")
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_run_forever, name="market-data-refresh", daemon=True)
    _thread.start()
    logger.info("Started scheduled market refresh")


def stop_scheduler():
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=5)


def scheduler_status():
    with _state_lock:
        running = _state["running"]
        last_run = _state["last_run"]
    try:
        persisted = storage.latest_job_run(
            fallback_path=os.path.join(os.path.dirname(__file__), "job_runs.json")
        )
        last_successful = storage.latest_job_run(
            fallback_path=os.path.join(os.path.dirname(__file__), "job_runs.json"),
            successful_only=True,
        )
    except Exception as exc:
        persisted = None
        last_successful = None
        status_error = f"{type(exc).__name__}: {exc}"
    else:
        status_error = None
    return {
        "enabled": os.getenv("MARKET_DATA_REFRESH_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"},
        "running": running,
        "symbols": _symbols(),
        "refresh_minutes": max(5, int(os.getenv("MARKET_DATA_REFRESH_MINUTES", "360"))) if os.getenv("MARKET_DATA_REFRESH_MINUTES", "360").isdigit() else 360,
        "last_run": last_run or persisted,
        "last_successful_update": last_successful,
        "storage_error": status_error,
        "data_note": "Daily bars use the latest data Yahoo Finance makes available; updates may be delayed or unavailable on market holidays.",
    }
