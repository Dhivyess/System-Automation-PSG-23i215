from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.config import SAFETY_THRESHOLDS, JointThresholds, ArmStatus

@dataclass
class SafetyViolation:
    joint: str
    metric: str
    observed_value: float
    threshold_limit: float
    units: str
    severity: str
    message: str


@dataclass
class SafetyCheckResult:
    is_tripped: bool
    status: ArmStatus
    violation_count: int
    violations: List[SafetyViolation] = field(default_factory=list)
    execution_time_us: float = 0.0

    def summary(self) -> str:
        if not self.is_tripped:
            return "DETERMINISTIC_SAFETY_PASS: All joint readings within safe operating envelope."
        details = "; ".join([f"{v.joint.upper()} {v.metric}={v.observed_value}{v.units} (Limit: {v.threshold_limit}{v.units})" for v in self.violations])
        return f"EMERGENCY_STOP_TRIGGERED: Deterministic safety trip on {len(self.violations)} parameter(s): {details}"


class DeterministicSafetyGuard:
    """
    Deterministic Safety Monitor.
    Runs independent of any machine learning model or external network.
    Evaluation complexity: O(1) deterministic arithmetic comparisons.
    """

    def __init__(self, thresholds: Optional[Dict[str, JointThresholds]] = None):
        self.thresholds = thresholds or SAFETY_THRESHOLDS

    def evaluate(self, telemetry: Dict[str, float]) -> SafetyCheckResult:
       
        import time
        start_time = time.perf_counter_ns()
        violations: List[SafetyViolation] = []

        sh_thresh = self.thresholds["shoulder"]
        sh_torque = telemetry.get("shoulder_torque_nm", 0.0)
        sh_temp = telemetry.get("shoulder_temp_c", 0.0)
        sh_vib = telemetry.get("shoulder_vib_mms", 0.0)

        if sh_torque > sh_thresh.max_torque_nm:
            violations.append(SafetyViolation(
                joint="shoulder", metric="torque", observed_value=sh_torque,
                threshold_limit=sh_thresh.max_torque_nm, units="Nm", severity="CRITICAL",
                message=f"Shoulder torque {sh_torque:.2f} Nm exceeds mechanical yield limit {sh_thresh.max_torque_nm} Nm"
            ))
        if sh_temp > sh_thresh.max_temp_c:
            violations.append(SafetyViolation(
                joint="shoulder", metric="temperature", observed_value=sh_temp,
                threshold_limit=sh_thresh.max_temp_c, units="°C", severity="CRITICAL",
                message=f"Shoulder temperature {sh_temp:.2f} °C exceeds motor insulation limit {sh_thresh.max_temp_c} °C"
            ))
        if sh_vib > sh_thresh.max_vib_mms:
            violations.append(SafetyViolation(
                joint="shoulder", metric="vibration", observed_value=sh_vib,
                threshold_limit=sh_thresh.max_vib_mms, units="mm/s", severity="CRITICAL",
                message=f"Shoulder vibration {sh_vib:.2f} mm/s exceeds ISO 20816 Zone D trip limit {sh_thresh.max_vib_mms} mm/s"
            ))

        el_thresh = self.thresholds["elbow"]
        el_torque = telemetry.get("elbow_torque_nm", 0.0)
        el_temp = telemetry.get("elbow_temp_c", 0.0)
        el_vib = telemetry.get("elbow_vib_mms", 0.0)

        if el_torque > el_thresh.max_torque_nm:
            violations.append(SafetyViolation(
                joint="elbow", metric="torque", observed_value=el_torque,
                threshold_limit=el_thresh.max_torque_nm, units="Nm", severity="CRITICAL",
                message=f"Elbow torque {el_torque:.2f} Nm exceeds gear shear limit {el_thresh.max_torque_nm} Nm"
            ))
        if el_temp > el_thresh.max_temp_c:
            violations.append(SafetyViolation(
                joint="elbow", metric="temperature", observed_value=el_temp,
                threshold_limit=el_thresh.max_temp_c, units="°C", severity="CRITICAL",
                message=f"Elbow temperature {el_temp:.2f} °C exceeds motor thermal limit {el_thresh.max_temp_c} °C"
            ))
        if el_vib > el_thresh.max_vib_mms:
            violations.append(SafetyViolation(
                joint="elbow", metric="vibration", observed_value=el_vib,
                threshold_limit=el_thresh.max_vib_mms, units="mm/s", severity="CRITICAL",
                message=f"Elbow vibration {el_vib:.2f} mm/s exceeds destructive resonance limit {el_thresh.max_vib_mms} mm/s"
            ))

        wr_thresh = self.thresholds["wrist"]
        wr_torque = telemetry.get("wrist_torque_nm", 0.0)
        wr_temp = telemetry.get("wrist_temp_c", 0.0)
        wr_vib = telemetry.get("wrist_vib_mms", 0.0)

        if wr_torque > wr_thresh.max_torque_nm:
            violations.append(SafetyViolation(
                joint="wrist", metric="torque", observed_value=wr_torque,
                threshold_limit=wr_thresh.max_torque_nm, units="Nm", severity="CRITICAL",
                message=f"Wrist torque {wr_torque:.2f} Nm exceeds bearing shear limit {wr_thresh.max_torque_nm} Nm"
            ))
        if wr_temp > wr_thresh.max_temp_c:
            violations.append(SafetyViolation(
                joint="wrist", metric="temperature", observed_value=wr_temp,
                threshold_limit=wr_thresh.max_temp_c, units="°C", severity="CRITICAL",
                message=f"Wrist temperature {wr_temp:.2f} °C exceeds servomotor thermal rating {wr_thresh.max_temp_c} °C"
            ))
        if wr_vib > wr_thresh.max_vib_mms:
            violations.append(SafetyViolation(
                joint="wrist", metric="vibration", observed_value=wr_vib,
                threshold_limit=wr_thresh.max_vib_mms, units="mm/s", severity="CRITICAL",
                message=f"Wrist vibration {wr_vib:.2f} mm/s exceeds bearing race destruction limit {wr_thresh.max_vib_mms} mm/s"
            ))

        elapsed_us = (time.perf_counter_ns() - start_time) / 1000.0
        is_tripped = len(violations) > 0

        return SafetyCheckResult(
            is_tripped=is_tripped,
            status=ArmStatus.EMERGENCY_STOP if is_tripped else ArmStatus.SAFE,
            violation_count=len(violations),
            violations=violations,
            execution_time_us=round(elapsed_us, 2)
        )
