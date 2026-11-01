"""Explainable, display-only PLC actions for the hackathon demonstration."""
from __future__ import annotations

from collections.abc import Sequence

from server.app.schemas import Detection, SimulatedPLC, Verdict


def simulated_plc_action(
    detections: Sequence[Detection], verdict: Verdict, severity: int
) -> SimulatedPLC:
    """Map a completed inspection to a demo action; never transmit it."""
    if verdict == "fail" and severity >= 4:
        return SimulatedPLC(action="STOP", target="Hold line for inspection", reason="Critical defect severity")
    if verdict == "fail" and any(
        "scratch" in d.class_name.lower() and d.conf >= 0.80 for d in detections
    ):
        return SimulatedPLC(action="MARK", target="Hold part for review", reason="High-confidence scratch")
    if verdict == "fail" and severity >= 3:
        return SimulatedPLC(action="DIVERT", target="Quality inspection lane", reason="Major defect severity")
    if verdict == "fail" or verdict == "review":
        return SimulatedPLC(action="MARK", target="Hold part for review", reason="Defect or uncertain result")
    return SimulatedPLC(action="LOG", target="Release part", reason="No actionable defect")
