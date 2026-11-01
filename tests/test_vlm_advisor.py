import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from server.app.vlm_advisor import LocalVLM, rule_route


class VLMAdvisorTests(unittest.TestCase):
    def test_visual_policy_uses_defects_not_confidence_as_damage(self):
        self.assertEqual(rule_route("steel", [])[0], "reuse_candidate")
        self.assertEqual(rule_route("transistor", [], uncertain_identity=True)[0], "inspect_or_repair")
        self.assertEqual(rule_route("steel", [{"class": "scratches", "conf": 0.99, "severity": 2}])[0], "inspect_or_repair")
        self.assertEqual(rule_route("bottle", [{"class": "broken_large", "conf": 0.5, "severity": 2}])[0], "recycle_candidate")
        self.assertEqual(rule_route("screw", [{"class": "thread_top", "conf": 0.7, "severity": 4}])[0], "recycle_candidate")

    def test_structured_local_response_and_disagreement(self):
        model = LocalVLM(SimpleNamespace(vlm_url="http://127.0.0.1:11434", vlm_model="qwen2.5vl:3b"))
        reply = {"visible_condition": "A line is visible on the steel surface.", "evidence": ["thin line"], "suggested_route": "recycle_candidate", "uncertainty": "Depth is unknown."}
        response = Mock()
        response.json.return_value = {"message": {"content": json.dumps(reply)}}
        response.raise_for_status.return_value = None
        image = Path("artifacts/demo_images/recommended/steel_patches.jpg")
        with patch("server.app.vlm_advisor.requests.post", return_value=response) as send:
            result = model.advise(image, "steel", [{"class": "patches", "conf": 0.9, "severity": 2}], "insp-test")
        self.assertEqual(result["recommendation"], "inspect_or_repair")
        self.assertTrue(result["disagreement"])
        self.assertTrue(result["advisory_only"])
        payload = send.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "qwen2.5vl:3b")
        self.assertTrue(payload["messages"][0]["images"])


if __name__ == "__main__":
    unittest.main()
