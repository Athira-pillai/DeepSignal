import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from sklearn.ensemble import RandomForestClassifier

from backend import model_registry


class ModelRegistryTests(unittest.TestCase):
    def test_artifact_reloads_in_a_clean_process_and_keeps_prediction_schema(self):
        features = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
        labels = [0, 1, 0, 1]
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(features, labels)
        identity = {
            "symbol": "TEST", "data_sha256": "known-test-data",
            "features": ["feature_a", "feature_b"], "seed": 42,
        }
        version = model_registry.version_for(identity)
        metadata = {
            "symbol": "TEST", "data": {"sha256": "known-test-data"},
            "features": {"random_forest_ordered": ["feature_a", "feature_b"]},
            "model_statuses": {"random_forest": "trained"},
        }
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"DEEPSIGNAL_ARTIFACT_DIR": directory}):
                model_registry.save_version(
                    version, identity,
                    {"models": {"random_forest": model}, "statuses": {"random_forest": "trained"}},
                    metadata,
                )
            script = (
                "import json; from backend import model_registry; "
                f"bundle=model_registry.load_version({version!r}); "
                "model=bundle['models']['random_forest']; "
                "result=model.predict_proba([[0.0, 1.0]])[0].tolist(); "
                "print(json.dumps({'model_version': bundle['metadata']['model_version'], 'probabilities': result}))"
            )
            child_env = os.environ.copy()
            child_env["DEEPSIGNAL_ARTIFACT_DIR"] = directory
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                env=child_env, check=True, capture_output=True, text=True,
            )

        output = json.loads(completed.stdout)
        self.assertEqual(output["model_version"], version)
        self.assertEqual(len(output["probabilities"]), 2)
        self.assertAlmostEqual(sum(output["probabilities"]), 1.0)

    def test_version_changes_when_training_identity_changes(self):
        original = {"symbol": "TEST", "data_sha256": "first"}
        changed = {"symbol": "TEST", "data_sha256": "second"}
        self.assertNotEqual(model_registry.version_for(original), model_registry.version_for(changed))


if __name__ == "__main__":
    unittest.main()
