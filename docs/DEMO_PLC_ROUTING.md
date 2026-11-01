# Simulated PLC routing for the hackathon

The live inspection screen displays an advisory PLC-style command. It is computed from the existing fusion verdict and severity. No command is sent to a PLC or other equipment.

Rules are evaluated in order:

| Inspection result | Displayed command | Displayed target |
| --- | --- | --- |
| FAIL and severity 4 | STOP | Hold line for inspection |
| FAIL with a scratch detection at confidence ≥ 0.80 | MARK | Hold part for review |
| FAIL and severity 3 | DIVERT | Quality inspection lane |
| Other FAIL or REVIEW | MARK | Hold part for review |
| PASS | LOG | Release part |

The critical rule takes priority over the scratch rule. A scratch at severity 4 therefore displays STOP. The command is included in each live frame as `simulated_plc` with `action`, `target`, `reason`, and `demo_only: true`. Severity is shown on a 1–4 scale. Confidence is the detector's score, not damaged surface area.
