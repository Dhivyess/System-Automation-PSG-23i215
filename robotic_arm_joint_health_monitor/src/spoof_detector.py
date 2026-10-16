import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.config import SPOOF_CONFIG, SpoofDefenseConfig


@dataclass
class SpoofCheckResult:
    is_spoofed: bool
    attack_type: Optional[str]
    flagged_joints: List[str]
    confidence: float
    forensic_details: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.is_spoofed:
            return "SPOOF_DEFENSE_PASS: Telemetry satisfies physical coupling and kinematic laws."
        joints_str = ", ".join([j.upper() for j in self.flagged_joints])
        return (
            f"SECURITY_ALERT: SENSOR_SPOOFING_DETECTED [{self.attack_type}] on {joints_str} "
            f"(Confidence: {self.confidence:.0%}). {'; '.join(self.forensic_details)}"
        )


class SpoofDefenseValidator:
    def __init__(self, config: Optional[SpoofDefenseConfig] = None):
        self.config = config or SPOOF_CONFIG

    def validate(
        self,
        telemetry: Dict[str, float],
        history_window: Optional[List[Dict[str, float]]] = None
    ) -> SpoofCheckResult:

        forensics: List[str] = []
        flagged_joints: List[str] = []
        detected_attacks: List[str] = []

        joints = ["shoulder", "elbow", "wrist"]
        for j in joints:
            torque = telemetry.get(f"{j}_torque_nm", 0.0)
            temp = telemetry.get(f"{j}_temp_c", 0.0)
            vib = telemetry.get(f"{j}_vib_mms", 0.0)
            min_torque_under_stress = self.config.min_physical_torque_under_high_stress.get(j, 20.0)
            if (vib >= self.config.high_vib_trigger_mms) and (temp >= self.config.high_temp_trigger_c):
                if torque < min_torque_under_stress:
                    flagged_joints.append(j)
                    detected_attacks.append("TORQUE_MASKING_THERMODYNAMIC_VIOLATION")
                    forensics.append(
                        f"{j.upper()} joint exhibits severe physical distress (Vib={vib:.2f} mm/s, "
                        f"Temp={temp:.2f} °C) but reported torque is suspiciously nominal ({torque:.2f} Nm < "
                        f"physical minimum threshold {min_torque_under_stress:.2f} Nm). "
                        "Violates thermodynamic energy dissipation law."
                    )
        sh_torque = telemetry.get("shoulder_torque_nm", 0.0)
        el_torque = telemetry.get("elbow_torque_nm", 0.0)
        wr_torque = telemetry.get("wrist_torque_nm", 0.0)

        distal_load = el_torque + (wr_torque * 1.5)  
        if distal_load > 55.0 and sh_torque < (distal_load * self.config.kinematic_shoulder_min_ratio):
            if "shoulder" not in flagged_joints:
                flagged_joints.append("shoulder")
                detected_attacks.append("KINEMATIC_CHAIN_REACTION_VIOLATION")
                forensics.append(
                    f"Shoulder torque ({sh_torque:.2f} Nm) is implausibly low relative to downstream dynamic load "
                    f"(Elbow={el_torque:.2f} Nm, Wrist={wr_torque:.2f} Nm). Violates serial kinematic equilibrium."
                )

        if history_window and len(history_window) >= 5:
            for j in joints:
                t_key = f"{j}_torque_nm"
                recent_torques = [frame.get(t_key, 0.0) for frame in history_window]
                var_torque = float(np.var(recent_torques))
                if var_torque < self.config.frozen_variance_epsilon:
                    flagged_joints.append(j)
                    detected_attacks.append("STATIC_REPLAY_FROZEN_TORQUE")
                    forensics.append(
                        f"{j.upper()} torque signal is static across {len(history_window)} cycles "
                        f"(variance = {var_torque:.6f}). Real analog strain gauge sensor noise absent."
                    )

        is_spoofed = len(flagged_joints) > 0
        attack_type = " / ".join(list(dict.fromkeys(detected_attacks))) if is_spoofed else None
        confidence = 0.95 if is_spoofed else 0.0

        return SpoofCheckResult(
            is_spoofed=is_spoofed,
            attack_type=attack_type,
            flagged_joints=list(dict.fromkeys(flagged_joints)),
            confidence=confidence,
            forensic_details=forensics
        )
