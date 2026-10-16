from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.config import ArmStatus
from src.deterministic_safety import SafetyCheckResult
from src.spoof_detector import SpoofCheckResult
from src.wear_predictor import WearPredictionResult
from src.log_parser import ArmLogSummary


@dataclass
class ArmDiagnosticReport:
    arm_id: str
    final_status: ArmStatus
    primary_reason: str
    detailed_justification: str
    recommended_action: str
    fused_risk_score: float            
    deterministic_result: SafetyCheckResult
    spoof_result: SpoofCheckResult
    ml_prediction: Optional[WearPredictionResult]
    log_summary: ArmLogSummary
    telemetry_snapshot: Dict[str, float] = field(default_factory=dict)


class HealthDecisionEngine:
   
    def __init__(self, wear_weight: float = 0.60, log_weight: float = 0.40):
        self.wear_weight = wear_weight
        self.log_weight = log_weight

    def evaluate_arm(
        self,
        arm_id: str,
        telemetry: Dict[str, float],
        deterministic_result: SafetyCheckResult,
        spoof_result: SpoofCheckResult,
        ml_prediction: Optional[WearPredictionResult],
        log_summary: ArmLogSummary
    ) -> ArmDiagnosticReport:
        if deterministic_result.is_tripped:
            violations_str = "; ".join([v.message for v in deterministic_result.violations])
            return ArmDiagnosticReport(
                arm_id=arm_id,
                final_status=ArmStatus.EMERGENCY_STOP,
                primary_reason="HARDWARE_SAFETY_LIMIT_BREACHED",
                detailed_justification=(
                    f"Immediate deterministic safety threshold exceeded on {deterministic_result.violation_count} "
                    f"parameter(s). Rule engine triggered instant shutdown without ML dependency. {violations_str}"
                ),
                recommended_action="IMMEDIATE E-STOP INTERLOCK ENGAGED. Lock out power and dispatch field engineer.",
                fused_risk_score=1.0,
                deterministic_result=deterministic_result,
                spoof_result=spoof_result,
                ml_prediction=ml_prediction,
                log_summary=log_summary,
                telemetry_snapshot=telemetry
            )
        if spoof_result.is_spoofed:
            return ArmDiagnosticReport(
                arm_id=arm_id,
                final_status=ArmStatus.EMERGENCY_STOP,
                primary_reason=f"SECURITY_ATTACK_DETECTED_{spoof_result.attack_type}",
                detailed_justification=(
                    f"Sensor spoofing attack detected ({spoof_result.attack_type}) on "
                    f"{', '.join(spoof_result.flagged_joints)}. Injected torque readings violate physical "
                    f"thermodynamic/kinematic conservation laws. Details: {'; '.join(spoof_result.forensic_details)}"
                ),
                recommended_action="SECURITY EMERGENCY STOP. Quarantine robot telemetry channel; inspect physical actuator for hidden damage.",
                fused_risk_score=1.0,
                deterministic_result=deterministic_result,
                spoof_result=spoof_result,
                ml_prediction=ml_prediction,
                log_summary=log_summary,
                telemetry_snapshot=telemetry
            )

        wear_score = ml_prediction.wear_risk_score if ml_prediction else 0.0
        log_score = log_summary.max_severity_score

        fused_score = (self.wear_weight * wear_score) + (self.log_weight * log_score)
        fused_score = round(min(1.0, max(0.0, fused_score)), 3)

        is_maintenance = False
        reasons: List[str] = []

        if ml_prediction and ml_prediction.predicted_label == "CRITICAL":
            is_maintenance = True
            reasons.append(f"ML predictive model flagged CRITICAL wear risk ({wear_score:.1%}) on {ml_prediction.most_worn_joint} joint")

        if log_summary.overall_severity == "HIGH":
            is_maintenance = True
            reasons.append(f"Technician inspection log reported HIGH severity mechanical defects ({', '.join(log_summary.summary_findings[:2])})")

        if wear_score >= 0.50 and log_summary.overall_severity in ["MEDIUM", "HIGH"]:
            is_maintenance = True
            reasons.append("Cross-validation agreement: ML wear prediction confirmed by technician physical observations")

        if fused_score >= 0.58:
            is_maintenance = True
            reasons.append(f"Fused risk score ({fused_score:.2f}) exceeded maintenance scheduling threshold (0.58)")

        if is_maintenance:
            return ArmDiagnosticReport(
                arm_id=arm_id,
                final_status=ArmStatus.SCHEDULE_MAINTENANCE,
                primary_reason="PREDICTIVE_WEAR_AND_OBSERVATIONAL_DEGRADATION",
                detailed_justification="; ".join(reasons),
                recommended_action="Schedule maintenance within next 24-48 hours. Inspect gearbox backlash, bearings, and replenish lubrication.",
                fused_risk_score=fused_score,
                deterministic_result=deterministic_result,
                spoof_result=spoof_result,
                ml_prediction=ml_prediction,
                log_summary=log_summary,
                telemetry_snapshot=telemetry
            )

        is_monitor = False
        monitor_reasons: List[str] = []

        if ml_prediction and ml_prediction.predicted_label == "DEGRADED":
            is_monitor = True
            monitor_reasons.append(f"ML predictive model detected incipient degradation (Wear Score: {wear_score:.1%}) on {ml_prediction.most_worn_joint}")

        if log_summary.overall_severity in ["LOW", "MEDIUM"]:
            is_monitor = True
            monitor_reasons.append(f"Technician recorded non-critical mechanical observations ({log_summary.overall_severity} severity)")

        if fused_score >= 0.25:
            is_monitor = True
            monitor_reasons.append(f"Fused health risk index ({fused_score:.2f}) indicates elevated wear drift")

        if is_monitor:
            return ArmDiagnosticReport(
                arm_id=arm_id,
                final_status=ArmStatus.MONITOR,
                primary_reason="INCIPIENT_WEAR_OR_MILD_OBSERVATION",
                detailed_justification="; ".join(monitor_reasons),
                recommended_action="Place arm on high-frequency monitoring. Log vibration trends during high-payload shifts.",
                fused_risk_score=fused_score,
                deterministic_result=deterministic_result,
                spoof_result=spoof_result,
                ml_prediction=ml_prediction,
                log_summary=log_summary,
                telemetry_snapshot=telemetry
            )

        return ArmDiagnosticReport(
            arm_id=arm_id,
            final_status=ArmStatus.SAFE,
            primary_reason="NOMINAL_OPERATIONS",
            detailed_justification=(
                f"All joint metrics within normal operating bounds. ML wear model confirms baseline health "
                f"(Wear Score: {wear_score:.1%}). Maintenance logs contain no mechanical defects."
            ),
            recommended_action="Normal production operation approved. Standard inspection schedule maintained.",
            fused_risk_score=fused_score,
            deterministic_result=deterministic_result,
            spoof_result=spoof_result,
            ml_prediction=ml_prediction,
            log_summary=log_summary,
            telemetry_snapshot=telemetry
        )
