import unittest

from server.app.demo_routing import simulated_plc_action
from server.app.schemas import Detection


def defect(name: str, confidence: float) -> Detection:
    return Detection(id=name, class_name=name, conf=confidence, bbox=[0, 0, 20, 20])


class DemoRoutingTests(unittest.TestCase):
    def test_critical_takes_priority_over_scratch(self):
        result = simulated_plc_action([defect("scratches", 0.95)], "fail", 4)
        self.assertEqual(result.action, "STOP")
        self.assertTrue(result.demo_only)

    def test_high_confidence_scratch_is_marked_for_review(self):
        result = simulated_plc_action([defect("scratch_head", 0.85)], "fail", 2)
        self.assertEqual((result.action, result.target), ("MARK", "Hold part for review"))

    def test_major_defect_is_diverted(self):
        self.assertEqual(simulated_plc_action([defect("inclusion", 0.7)], "fail", 3).action, "DIVERT")

    def test_uncertain_result_is_marked(self):
        self.assertEqual(simulated_plc_action([], "review", 2).action, "MARK")

    def test_pass_is_logged(self):
        self.assertEqual(simulated_plc_action([], "pass", 1).action, "LOG")


if __name__ == "__main__":
    unittest.main()
