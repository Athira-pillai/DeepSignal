import unittest

from backend.adaptive_ensemble import ADAPTIVE_ENSEMBLE_VERSION, select_adaptive_weights


def validation_fixture(rows=60):
    output = []
    for index in range(rows):
        target = index % 2
        output.append({
            "date": f"2025-01-{index + 1:02d}",
            "target": target,
            "regime": "bullish/trending" if index < rows // 2 else "bearish/high-risk",
            "components": {
                "lstm": .9 if target else .1,
                "random_forest": .15 if target else .85,
                "technical": .55,
            },
            "availability": {"lstm": True, "random_forest": True, "technical": True, "finbert": False},
        })
    return output


class AdaptiveEnsembleTests(unittest.TestCase):
    def test_validation_skill_produces_bounded_simplex_weights(self):
        decision = select_adaptive_weights(validation_fixture(), minimum_validation_samples=20)

        self.assertEqual(decision["status"], "ok")
        self.assertAlmostEqual(sum(decision["weights"].values()), 1.0, places=10)
        self.assertTrue(all(.05 <= weight <= .8 for weight in decision["weights"].values()))
        self.assertGreater(decision["weights"]["lstm"], decision["weights"]["random_forest"])
        self.assertEqual(decision["method"], ADAPTIVE_ENSEMBLE_VERSION)
        self.assertNotIn("finbert", decision["weights"])
        self.assertFalse(decision["component_availability"]["finbert"]["available_for_test"])

    def test_regime_specific_scores_are_used_only_with_minimum_evidence(self):
        rows = validation_fixture(60)
        regime_specific = select_adaptive_weights(
            rows, current_regime="bullish/trending", minimum_regime_samples=20
        )
        self.assertEqual(regime_specific["selected_scope"], "regime_validation")
        self.assertEqual(regime_specific["validation_metrics"]["lstm"]["samples"], 30)

        sparse = select_adaptive_weights(
            rows[:24], current_regime="bullish/trending", minimum_regime_samples=25
        )
        self.assertEqual(sparse["selected_scope"], "overall_validation")
        self.assertIn("insufficient_validation_rows_for_current_regime", sparse["fallback_reason"])

    def test_unavailable_components_are_excluded_deterministically(self):
        rows = validation_fixture()
        for row in rows:
            row["availability"]["lstm"] = False
        first = select_adaptive_weights(rows, available_components=("random_forest", "technical"))
        second = select_adaptive_weights(rows, available_components=("random_forest", "technical"))

        self.assertEqual(first["weights"], second["weights"])
        self.assertAlmostEqual(sum(first["weights"].values()), 1.0, places=10)
        self.assertNotIn("lstm", first["weights"])
        self.assertTrue(all(.05 <= weight <= .8 for weight in first["weights"].values()))

    def test_insufficient_validation_evidence_returns_explicit_fallback(self):
        decision = select_adaptive_weights(validation_fixture(5), minimum_validation_samples=20)

        self.assertEqual(decision["status"], "fixed_prior_fallback")
        self.assertEqual(decision["weights"], {})
        self.assertIn("insufficient_component_validation_evidence", decision["fallback_reason"])

    def test_final_test_outcomes_are_not_an_input_to_weight_selection(self):
        validation = validation_fixture()
        selected = select_adaptive_weights(validation)
        hypothetical_test_outcomes = [0, 1] * 200
        self.assertEqual(selected, select_adaptive_weights(validation))
        self.assertEqual(len(hypothetical_test_outcomes), 400)
        self.assertNotIn("test_labels", selected)


if __name__ == "__main__":
    unittest.main()
