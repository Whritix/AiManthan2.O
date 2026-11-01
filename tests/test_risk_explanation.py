import unittest

from server.app.risk_explanation import IMPACTS, explain_flagged_risk


class RiskExplanationTests(unittest.TestCase):
    def test_all_trained_classes_have_scenarios(self):
        self.assertEqual(len(IMPACTS), 21)

    def test_flagged_screw_has_possible_effect_without_fabricated_lifetime(self):
        result = explain_flagged_risk("screw", [{"class": "thread_side", "conf": 0.8, "severity": 3}], "fail")
        self.assertEqual(result["defect"], "thread_side")
        self.assertIn("loosen", result["if_used"])
        self.assertEqual(result["time_to_failure"], "Cannot estimate from an image alone.")

    def test_pass_does_not_show_consequence(self):
        self.assertIsNone(explain_flagged_risk("steel", [{"class": "scratches", "severity": 1}], "pass"))
        self.assertIsNone(explain_flagged_risk("transistor", [], "review"))


if __name__ == "__main__":
    unittest.main()
