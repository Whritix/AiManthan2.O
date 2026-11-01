import unittest
from pathlib import Path

import numpy as np

from server.app.auto_profile import AutoProfileClassifier, confirm_near_miss


class FakeModel:
    classes_ = np.array(["steel", "other"])

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, features):
        return np.array([self.probabilities], dtype=float)


class AutoProfileScopeTests(unittest.TestCase):
    def identify(self, probabilities):
        classifier = AutoProfileClassifier(Path("missing-model.joblib"))
        classifier._model = FakeModel(probabilities)
        return classifier.identify(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_supported_family_is_accepted(self):
        self.assertEqual(self.identify([0.95, 0.05])["profile"], "steel")

    def test_other_class_is_rejected(self):
        result = self.identify([0.05, 0.95])
        self.assertEqual(result["status"], "unsupported")
        self.assertIsNone(result["profile"])

    def test_nonfinite_prediction_is_rejected(self):
        self.assertEqual(self.identify([float("nan"), 0.5])["status"], "uncertain")

    def test_narrow_vlm_agreement_confirms_family(self):
        result = confirm_near_miss(
            {"status": "uncertain", "candidate": "transistor", "confidence": 0.517, "profile": None},
            {"product": "transistor", "visual_evidence": ["black body", "three leads"], "uncertainty": ""},
        )
        self.assertEqual(result["profile"], "transistor")
        self.assertEqual(result["method"], "vlm_confirmed")
        self.assertIn("Review", result["domain_warning"])

    def test_vlm_disagreement_or_weak_classifier_stays_uncertain(self):
        identification = {"status": "uncertain", "candidate": "transistor", "confidence": 0.517, "profile": None}
        other = {"product": "other", "visual_evidence": ["round body", "three leads"], "uncertainty": ""}
        self.assertEqual(confirm_near_miss(identification, other), identification)
        weak = {**identification, "confidence": 0.30}
        self.assertEqual(confirm_near_miss(weak, {**other, "product": "transistor"}), weak)


if __name__ == "__main__":
    unittest.main()
