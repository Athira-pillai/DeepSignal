"""Local, versioned model artifacts with atomic writes and safe loading."""

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import tempfile
import threading
from collections import OrderedDict
from datetime import datetime, timezone


ARTIFACT_FORMAT_VERSION = 1
_VERSION_PATTERN = re.compile(r"^deepsignal-model-v1-[a-f0-9]{20}$")
_write_lock = threading.RLock()
_loaded_lock = threading.RLock()
_loaded_versions = OrderedDict()
_MAX_LOADED_VERSIONS = 8


class ModelArtifactError(RuntimeError):
    pass


def artifact_root():
    configured = os.getenv("DEEPSIGNAL_ARTIFACT_DIR", "").strip()
    return os.path.abspath(configured) if configured else os.path.join(os.path.dirname(__file__), "model_artifacts")


def package_versions():
    names = ("numpy", "pandas", "scikit-learn", "tensorflow", "xgboost", "transformers")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def version_for(identity):
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"deepsignal-model-v1-{digest}"


def _version_directory(version):
    if not isinstance(version, str) or not _VERSION_PATTERN.fullmatch(version):
        raise ModelArtifactError("Invalid model version identifier.")
    return os.path.join(artifact_root(), version)


def has_version(version):
    return os.path.isfile(os.path.join(_version_directory(version), "metadata.json"))


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)


def save_version(version, identity, captured, metadata):
    """Write metadata and all fitted preprocessing/model objects as one version."""
    destination = _version_directory(version)
    root = artifact_root()
    os.makedirs(root, exist_ok=True)
    with _write_lock:
        if os.path.isfile(os.path.join(destination, "metadata.json")):
            return read_metadata(version)
        temporary = tempfile.mkdtemp(prefix="artifact-", dir=root)
        try:
            import joblib

            artifacts = {}
            for name, estimator in captured.get("models", {}).items():
                filename = f"{name}.joblib"
                joblib.dump(estimator, os.path.join(temporary, filename), compress=3)
                artifacts[name] = filename
            for name, preprocessor in captured.get("preprocessors", {}).items():
                filename = f"{name}.joblib"
                joblib.dump(preprocessor, os.path.join(temporary, filename), compress=3)
                artifacts[name] = filename
            lstm_model = captured.get("keras_models", {}).get("lstm")
            if lstm_model is not None:
                filename = "lstm.keras"
                lstm_model.save(os.path.join(temporary, filename))
                artifacts["lstm"] = filename

            record = {
                **metadata,
                "artifact_format_version": ARTIFACT_FORMAT_VERSION,
                "model_version": version,
                "identity": identity,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "artifact_location": os.path.relpath(destination, root).replace(os.sep, "/"),
                "artifacts": artifacts,
                "model_statuses": captured.get("statuses", {}),
            }
            _write_json(os.path.join(temporary, "metadata.json"), record)
            try:
                os.replace(temporary, destination)
            except OSError:
                # Another request may have stored this deterministic version first.
                if os.path.isfile(os.path.join(destination, "metadata.json")):
                    return read_metadata(version)
                raise
            return record
        except Exception as exc:
            if isinstance(exc, ModelArtifactError):
                raise
            raise ModelArtifactError(f"Could not save model artifacts ({type(exc).__name__}).") from exc
        finally:
            if os.path.isdir(temporary):
                shutil.rmtree(temporary, ignore_errors=True)


def read_metadata(version):
    path = os.path.join(_version_directory(version), "metadata.json")
    try:
        with open(path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ModelArtifactError("Model metadata is missing or unreadable.") from exc
    if metadata.get("model_version") != version or metadata.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise ModelArtifactError("Model metadata version or artifact format does not match.")
    if version_for(metadata.get("identity")) != version:
        raise ModelArtifactError("Model metadata identity does not match its version identifier.")
    return metadata


def load_version(version):
    """Load a version's estimator, scaler and optional Keras model into memory."""
    with _loaded_lock:
        if version in _loaded_versions:
            _loaded_versions.move_to_end(version)
            return _loaded_versions[version]
    metadata = read_metadata(version)
    directory = _version_directory(version)
    try:
        import joblib

        models, preprocessors = {}, {}
        for name, filename in metadata.get("artifacts", {}).items():
            path = os.path.join(directory, filename)
            if os.path.commonpath((directory, os.path.abspath(path))) != directory or not os.path.isfile(path):
                raise ModelArtifactError(f"Model artifact is missing or has an invalid path: {name}.")
            if name == "lstm":
                from tensorflow.keras.models import load_model
                models[name] = load_model(path, compile=False)
            elif name.endswith("_scaler"):
                preprocessors[name] = joblib.load(path)
            else:
                models[name] = joblib.load(path)
        statuses = metadata.get("model_statuses", {})
        for name, status in statuses.items():
            if status == "trained" and name not in models:
                raise ModelArtifactError(f"The trained {name} model artifact is missing.")
        if statuses.get("lstm") == "trained" and "lstm_scaler" not in preprocessors:
            raise ModelArtifactError("The trained LSTM preprocessing artifact is missing.")
        if "random_forest" in models:
            expected = metadata.get("features", {}).get("random_forest_ordered")
            actual = list(getattr(models["random_forest"], "feature_names_in_", []))
            if expected and actual and actual != expected:
                raise ModelArtifactError("Saved Random Forest feature order does not match its metadata.")
        if "lstm" in models:
            expected_shape = tuple(metadata.get("input_shapes", {}).get("lstm", []))
            actual_shape = tuple(models["lstm"].input_shape[1:])
            if expected_shape and actual_shape != expected_shape:
                raise ModelArtifactError("Saved LSTM input shape does not match its metadata.")
            expected_features = len(metadata.get("features", {}).get("lstm_ordered", []))
            scaler_features = getattr(preprocessors["lstm_scaler"], "n_features_in_", expected_features)
            if expected_features and scaler_features != expected_features:
                raise ModelArtifactError("Saved LSTM scaler feature count does not match its metadata.")
        bundle = {"metadata": metadata, "models": models, "preprocessors": preprocessors}
    except ModelArtifactError:
        raise
    except Exception as exc:
        raise ModelArtifactError(f"Could not load model artifacts ({type(exc).__name__}).") from exc
    with _loaded_lock:
        _loaded_versions[version] = bundle
        _loaded_versions.move_to_end(version)
        while len(_loaded_versions) > _MAX_LOADED_VERSIONS:
            _loaded_versions.popitem(last=False)
    return bundle


def list_versions(limit=50):
    root = artifact_root()
    if not os.path.isdir(root):
        return []
    versions = []
    for name in os.listdir(root):
        if not _VERSION_PATTERN.fullmatch(name):
            continue
        try:
            versions.append(read_metadata(name))
        except ModelArtifactError:
            continue
    versions.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return versions[:max(1, min(int(limit), 200))]
