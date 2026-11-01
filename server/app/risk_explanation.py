"""Conservative, class-specific failure scenarios for flagged demo inspections."""
from __future__ import annotations

from typing import Any


# These are possible consequences to investigate, not measured failure probabilities.
IMPACTS: dict[str, tuple[str, str]] = {
    "steel_crazing": ("Crack-like surface texture may make the strip unsuitable for the intended finish or forming step.", "If a true crack is present in a loaded part, it could grow during service; this image does not establish crack depth."),
    "steel_inclusion": ("An inclusion may interrupt the steel surface or local material continuity.", "Forming, machining, or loading could expose a local weakness, depending on inclusion size and location."),
    "steel_patches": ("The irregular patch may fail a surface-finish or coating requirement.", "Downstream coating or forming quality could be affected; structural damage cannot be concluded from the image."),
    "steel_pitted_surface": ("Pits may reduce usable surface quality and need a depth check.", "Deep pits in a stressed or exposed location could become local failure sites; depth is unknown."),
    "steel_scratches": ("A scratch may breach the required finish or protective coating.", "A deep scratch in a loaded area could become a local stress concentration; the image cannot measure depth."),
    "metal_nut_bent": ("A bent nut may not seat or engage the mating part correctly.", "Poor seating or engagement could lead to loosening or assembly damage under load."),
    "metal_nut_color": ("Discoloration may indicate a finish issue; its cause is not established visually.", "If it reflects coating damage or corrosion, fit or long-term protection could be affected."),
    "metal_nut_flip": ("The nut may be presented or assembled in the wrong orientation.", "Incorrect seating or engagement could cause assembly errors or a loose joint."),
    "metal_nut_scratch": ("A scratch may affect the nut's surface finish or contact face.", "If the contact or protective surface is damaged, joint performance may need checking."),
    "screw_manipulated_front": ("The screw front may not enter or seat as intended.", "Assembly could jam, damage the mating part, or leave a poor joint."),
    "screw_scratch_head": ("Head damage may interfere with the driver or required torque application.", "Insufficient tightening or a stripped drive could leave the joint unreliable."),
    "screw_scratch_neck": ("Damage near the neck may reduce the usable cross-section.", "Under load or repeated vibration, a severe notch could contribute to breakage."),
    "screw_thread_side": ("Side-thread damage may prevent proper engagement.", "The joint could strip, loosen, or fail to reach its specified clamp load."),
    "screw_thread_top": ("Top-thread damage may interfere with starting or engagement.", "Cross-threading, stripping, or insufficient clamp load could follow."),
    "transistor_bent_lead": ("A bent lead may not align with its intended connection.", "A poor joint or short could cause intermittent or failed circuit operation."),
    "transistor_cut_lead": ("A cut lead may prevent the intended electrical connection.", "The circuit may fail immediately or behave intermittently if contact is marginal."),
    "transistor_damaged_case": ("Case damage may compromise package protection.", "Electrical or thermal performance may be affected; bench tests are needed."),
    "transistor_misplaced": ("A misplaced component may not contact the intended pads or fixture points.", "Assembly or circuit operation could fail, including an unintended connection."),
    "bottle_broken_large": ("Broken edges can cut a handler and the bottle may not contain its contents.", "Leakage or contamination could occur during filling, transport, or use."),
    "bottle_broken_small": ("Even a small visible break can create a sharp edge or leak path.", "The damage could grow during handling and compromise containment."),
    "bottle_contamination": ("Visible contamination may make the bottle unsuitable for its intended contents.", "Contents could be contaminated if the bottle is filled or reused without a validated cleaning check."),
}

NEEDED_FOR_LIFE = {
    "steel": "Material grade, defect depth, service loads or cycles, environment, and inspection or test results.",
    "metal_nut": "Dimensions, material, thread engagement, torque, service loads, and joint tests.",
    "screw": "Dimensions, material, thread engagement, torque, service loads or cycles, and joint tests.",
    "transistor": "Electrical tests, operating voltage and current, temperature history, and package integrity checks.",
    "bottle": "Material, intended contents, handling conditions, and leak or impact tests.",
}


def explain_flagged_risk(profile: str, detections: list[dict[str, Any]], verdict: str) -> dict[str, Any] | None:
    """Return a potential consequence only for a named REVIEW/FAIL detection."""
    if verdict not in {"review", "fail"} or not detections:
        return None
    top = max(detections, key=lambda d: (int(d.get("severity") or 1), float(d.get("conf") or 0)))
    name = str(top.get("class") or top.get("class_name") or "")
    if name.startswith(f"{profile}_"):
        name = name[len(profile) + 1:]
    impact, possible_failure = IMPACTS.get(
        f"{profile}_{name}",
        ("The detected condition may violate the part's acceptance criteria.", "The downstream effect depends on the part's function and operating conditions."),
    )
    return {
        "product": profile,
        "defect": name,
        "decision": verdict,
        "severity": int(top.get("severity") or 1),
        "immediate_concern": impact,
        "if_used": possible_failure,
        "time_to_failure": "Cannot estimate from an image alone.",
        "needed_for_estimate": NEEDED_FOR_LIFE.get(profile, "Material, defect dimensions, operating conditions, and test data."),
        "note": "Possible failure scenario for review, not a diagnosis, failure probability, or safety certification.",
    }
