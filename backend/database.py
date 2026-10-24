"""Supabase REST storage. No database server or Python driver is installed locally."""

import json
import os
import tempfile
from datetime import datetime, timezone

import requests


class StorageError(RuntimeError):
    pass


def configuration_status():
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = os.getenv("SUPABASE_SECRET_KEY", "").strip()
    if not url and not key:
        return "not_configured"
    if not url or not key:
        return "incomplete"
    if not url.startswith("https://"):
        return "invalid_url"
    return "configured"


def _connection():
    status = configuration_status()
    if status != "configured":
        raise StorageError(f"Supabase storage is {status}; set SUPABASE_URL and SUPABASE_SECRET_KEY.")
    return os.getenv("SUPABASE_URL", "").strip().rstrip("/"), os.getenv("SUPABASE_SECRET_KEY", "").strip()


def _request(method, resource, *, params=None, body=None, prefer=None):
    url, key = _connection()
    headers = {
        "apikey": key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    try:
        response = requests.request(
            method,
            f"{url}/rest/v1/{resource}",
            params=params,
            json=body,
            headers=headers,
            timeout=15,
        )
    except requests.RequestException as exc:
        raise StorageError(f"Supabase request failed ({type(exc).__name__}).") from exc
    if not response.ok:
        detail = (response.text or "").replace("\n", " ")[:300]
        raise StorageError(f"Supabase returned HTTP {response.status_code}: {detail}")
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError as exc:
        raise StorageError("Supabase returned a response that was not valid JSON.") from exc


def _local_append(path, item):
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        try:
            with open(path, encoding="utf-8") as file:
                rows = json.load(file)
            if not isinstance(rows, list):
                rows = []
        except (OSError, ValueError):
            rows = []
        rows.append(item)
        fd, temporary = tempfile.mkstemp(prefix="storage-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(rows, file, indent=2)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    except OSError as exc:
        raise StorageError("Could not write the local JSON fallback record.") from exc


def save_prediction_bundle(prediction, fallback_path):
    status = configuration_status()
    if status == "not_configured":
        _local_append(fallback_path, {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "market_data": prediction.get("market_data", {}),
            "prediction": prediction,
            "signal": {
                "symbol": prediction.get("stock_symbol"),
                "signal_type": prediction.get("signal_type"),
                "strategy_version": prediction.get("signal_explanation", {}).get("hybrid_version"),
                "model_version": prediction.get("model_version"),
                "signal_timestamp": prediction.get("prediction_timestamp"),
                "reason": prediction.get("analysis_summary"),
                "risk_control_decision": None,
            },
        })
        return {"status": "saved_locally", "mode": "local_json"}
    if status != "configured":
        raise StorageError(f"Supabase storage is {status}; complete its environment settings.")

    market = prediction.get("market_data")
    explanation = prediction.get("signal_explanation", {})
    if not isinstance(market, dict) or not isinstance(explanation, dict) or "target" not in explanation:
        raise StorageError("Prediction is missing market or model metadata and was not saved.")
    market_body = {
        "symbol": prediction["stock_symbol"],
        "interval": market["interval"],
        "source": market["source"],
        "market_timestamp": market["market_timestamp"],
        "retrieved_at": market["retrieved_at"],
        "data_sha256": market["sha256"],
        "row_count": market["row_count"],
        "quality_status": market["quality_status"],
        "metadata": market,
    }
    markets = _request(
        "POST", "market_data_runs",
        params={"on_conflict": "symbol,interval,data_sha256"},
        body=market_body,
        prefer="resolution=merge-duplicates,return=representation",
    )
    if not isinstance(markets, list) or not markets:
        raise StorageError("Supabase did not return the stored market-data record.")
    prediction_body = {
        "symbol": prediction["stock_symbol"],
        "horizon": explanation["target"]["horizon_sessions"],
        "prediction_timestamp": prediction["prediction_timestamp"],
        "signal_type": prediction["signal_type"],
        "predicted_trend": prediction["predicted_trend"],
        "predicted_change": prediction["predicted_change"],
        "ensemble_probability_up": explanation["ensemble_probability_up"],
        "model_version": prediction.get("model_version") or explanation["hybrid_version"],
        "input_window_start": market["window_start"],
        "input_window_end": market["window_end"],
        "input_features": explanation,
        "market_data_run_id": markets[0]["id"],
        "response": {key: value for key, value in prediction.items() if key != "backtest"},
    }
    saved = _request("POST", "predictions", body=prediction_body, prefer="return=representation")
    if not isinstance(saved, list) or not saved:
        raise StorageError("Supabase did not return the stored prediction record.")
    signal_body = {
        "prediction_id": saved[0]["id"],
        "symbol": prediction["stock_symbol"],
        "signal_type": prediction["signal_type"],
        "strategy_version": explanation["hybrid_version"],
        "signal_timestamp": prediction["prediction_timestamp"],
        "reason": prediction["analysis_summary"],
        "risk_control_decision": None,
        "metadata": explanation,
    }
    _request("POST", "signals", body=signal_body, prefer="return=minimal")
    return {"status": "saved", "mode": "supabase", "prediction_id": saved[0]["id"]}


def save_experiment(report, experiment_type, fallback_path):
    if not report.get("run_id") or not report.get("symbol"):
        raise StorageError("Experiment report is missing its run ID or symbol and was not saved.")
    status = configuration_status()
    if status == "not_configured":
        _local_append(fallback_path, report)
        return {"status": "saved_locally", "mode": "local_json"}
    if status != "configured":
        raise StorageError(f"Supabase storage is {status}; complete its environment settings.")

    dates = report.get("data", {})
    split = report.get("split", {})
    versions = report.get("versions", {})
    body = {
        "run_id": report["run_id"],
        "experiment_type": experiment_type,
        "symbol": report.get("symbol"),
        "model_version": versions.get("hybrid_model") or versions.get("evaluation_model"),
        "strategy_version": report.get("strategy_version") or report.get("engine_version"),
        "window_start": split.get("test_start") or dates.get("start"),
        "window_end": split.get("test_end") or dates.get("end"),
        "data_sha256": dates.get("sha256"),
        "periods": split or dates,
        "metrics": report.get("models") or report.get("summary", {}),
        "parameters": report.get("configuration", {}),
        "assumptions": report.get("assumptions", {}),
        "report": report,
    }
    rows = _request(
        "POST", "experiments", params={"on_conflict": "run_id"}, body=body,
        prefer="resolution=merge-duplicates,return=representation",
    )
    if not isinstance(rows, list) or not rows:
        raise StorageError("Supabase did not return the stored experiment record.")
    return {"status": "saved", "mode": "supabase", "experiment_id": rows[0]["id"]}


def _read_local(path):
    try:
        with open(path, encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, list) else []
    except (OSError, ValueError):
        return []


def list_predictions(symbol=None, limit=50, fallback_path=None):
    if configuration_status() == "not_configured":
        rows = _read_local(fallback_path) if fallback_path else []
        if symbol:
            rows = [row for row in rows if row.get("prediction", {}).get("stock_symbol", "").upper() == symbol.upper()]
        return rows[-limit:][::-1]
    params = {"select": "id,symbol,horizon,prediction_timestamp,signal_type,model_version,response", "order": "prediction_timestamp.desc", "limit": limit}
    if symbol:
        params["symbol"] = f"eq.{symbol.strip().upper()}"
    rows = _request("GET", "predictions", params=params)
    return rows if isinstance(rows, list) else []


def list_experiments(experiment_type=None, limit=50, fallback_directory=None):
    if configuration_status() == "not_configured":
        if not fallback_directory:
            return []
        filenames = {
            "evaluation": ("evaluation_runs.json",),
            "backtest": ("backtest_runs.json",),
            None: ("evaluation_runs.json", "backtest_runs.json"),
        }[experiment_type]
        rows = []
        for filename in filenames:
            rows.extend(_read_local(os.path.join(fallback_directory, filename)))
        return rows[-limit:][::-1]
    params = {"select": "id,run_id,experiment_type,symbol,created_at,model_version,strategy_version,window_start,window_end,data_sha256,metrics,parameters,assumptions,report", "order": "created_at.desc", "limit": limit}
    if experiment_type:
        params["experiment_type"] = f"eq.{experiment_type}"
    rows = _request("GET", "experiments", params=params)
    return rows if isinstance(rows, list) else []


def check_connection():
    status = configuration_status()
    if status != "configured":
        return {"status": status, "persistent_storage": False}
    _request("GET", "experiments", params={"select": "id", "limit": 1})
    return {"status": "connected", "persistent_storage": True}


def save_market_bars(symbol, interval, rows, fallback_path=None):
    """Idempotently store validated OHLCV bars and their derived features."""
    status = configuration_status()
    normalized = [
        {"symbol": symbol, "interval": interval, **row}
        for row in rows
    ]
    if status == "not_configured":
        path = fallback_path or os.path.join(os.path.dirname(__file__), "market_data_bars.json")
        existing = _read_local(path)
        by_key = {
            (item.get("symbol"), item.get("interval"), item.get("timestamp")): item
            for item in existing
        }
        by_key.update({(item["symbol"], item["interval"], item["timestamp"]): item for item in normalized})
        try:
            directory = os.path.dirname(path) or "."
            os.makedirs(directory, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix="market-data-", suffix=".json", dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as file:
                    json.dump(list(by_key.values()), file, indent=2)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        except OSError as exc:
            raise StorageError("Could not write the local market-data fallback.") from exc
        return len(normalized)
    if status != "configured":
        raise StorageError(f"Supabase storage is {status}; complete its environment settings.")

    for offset in range(0, len(normalized), 500):
        _request(
            "POST", "market_data_bars",
            params={"on_conflict": "symbol,interval,timestamp"},
            body=normalized[offset:offset + 500],
            prefer="resolution=merge-duplicates,return=minimal",
        )
    return len(normalized)


def record_job_run(run, fallback_path=None):
    """Persist one background-job summary for inspection after a restart."""
    if configuration_status() == "not_configured":
        path = fallback_path or os.path.join(os.path.dirname(__file__), "job_runs.json")
        rows = _read_local(path)
        rows.append(run)
        _local_replace(path, rows[-100:])
        return
    if configuration_status() != "configured":
        raise StorageError(f"Supabase storage is {configuration_status()}; complete its environment settings.")
    _request("POST", "background_job_runs", body=run, prefer="return=minimal")


def latest_job_run(job_name="market_data_refresh", fallback_path=None, successful_only=False):
    if configuration_status() == "not_configured":
        path = fallback_path or os.path.join(os.path.dirname(__file__), "job_runs.json")
        rows = [row for row in _read_local(path) if row.get("job_name") == job_name]
        if successful_only:
            rows = [row for row in rows if row.get("status") in {"succeeded", "partial"}]
        return rows[-1] if rows else None
    params = {
        "select": "job_name,started_at,completed_at,status,rows_processed,details",
        "job_name": f"eq.{job_name}", "order": "completed_at.desc", "limit": 1,
    }
    if successful_only:
        params["status"] = "in.(succeeded,partial)"
    rows = _request(
        "GET", "background_job_runs",
        params=params,
    )
    return rows[0] if isinstance(rows, list) and rows else None


def _local_replace(path, value):
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="storage-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(value, file, indent=2)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    except OSError as exc:
        raise StorageError("Could not write the local JSON fallback record.") from exc


def migrate_local_experiments(directory):
    """Idempotently copy prior JSON experiment reports into Supabase on startup."""
    if configuration_status() != "configured":
        return 0
    moved = 0
    for filename, kind in (("evaluation_runs.json", "evaluation"), ("backtest_runs.json", "backtest")):
        path = os.path.join(directory, filename)
        try:
            with open(path, encoding="utf-8") as file:
                reports = json.load(file)
        except (OSError, ValueError):
            continue
        if not isinstance(reports, list):
            continue
        for report in reports:
            try:
                save_experiment(report, kind, path)
                moved += 1
            except StorageError as exc:
                # Keep the original file intact; a later restart can retry.
                return {"status": "partial", "imported": moved, "message": str(exc)}
            except (KeyError, TypeError):
                return {"status": "partial", "imported": moved, "message": "A local report was missing required fields."}
    return {"status": "imported", "imported": moved}
