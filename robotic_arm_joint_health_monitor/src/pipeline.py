from typing import Dict, List, Optional, Any
from pathlib import Path
from src.config import (
    ArmStatus,
    HISTORICAL_DATA_FILE,
    MAINTENANCE_LOG_FILE
)
from src.deterministic_safety import DeterministicSafetyGuard
from src.wear_predictor import PredictiveWearModel
from src.log_parser import MaintenanceLogParser
from src.spoof_detector import SpoofDefenseValidator
from src.decision_engine import HealthDecisionEngine, ArmDiagnosticReport


class RoboticArmMonitoringPipeline:
    def __init__(
        self,
        historical_data_path: Optional[Path] = None,
        maintenance_log_path: Optional[Path] = None,
        auto_train: bool = True
    ):
        self.safety_guard = DeterministicSafetyGuard()
        self.spoof_validator = SpoofDefenseValidator()
        self.wear_model = PredictiveWearModel()
        self.log_parser = MaintenanceLogParser(maintenance_log_path or MAINTENANCE_LOG_FILE)
        self.decision_engine = HealthDecisionEngine()

        if auto_train:
            self._initialize_wear_model(historical_data_path or HISTORICAL_DATA_FILE)

    def _initialize_wear_model(self, data_path: Path):
        if self.wear_model.model_path.exists():
            try:
                self.wear_model.load()
            except Exception:
                self.wear_model.train(data_path)
        else:
            self.wear_model.train(data_path)

    def evaluate_arm(
        self,
        arm_id: str,
        telemetry: Dict[str, float],
        ad_hoc_log_note: Optional[str] = None,
        history_window: Optional[List[Dict[str, float]]] = None
    ) -> ArmDiagnosticReport:
        safety_res = self.safety_guard.evaluate(telemetry)

        spoof_res = self.spoof_validator.validate(telemetry, history_window=history_window)

        ml_res = None
        try:
            ml_res = self.wear_model.predict(telemetry)
        except Exception as e:
            pass

        log_summary = self.log_parser.get_arm_summary(arm_id)
        if ad_hoc_log_note:
            parsed_ad_hoc = self.log_parser.parse_entry(f"[NOW] {arm_id} RealTimeTech: {ad_hoc_log_note}")
            if parsed_ad_hoc and parsed_ad_hoc.is_mechanically_relevant:
                if parsed_ad_hoc.severity_score > log_summary.max_severity_score:
                    log_summary.overall_severity = parsed_ad_hoc.severity
                    log_summary.max_severity_score = parsed_ad_hoc.severity_score
                for j in parsed_ad_hoc.mentioned_joints:
                    if j not in log_summary.affected_joints:
                        log_summary.affected_joints.append(j)
                log_summary.summary_findings.append(f"Ad-Hoc: {parsed_ad_hoc.raw_text}")
                log_summary.relevant_entries.append(parsed_ad_hoc)
                log_summary.has_relevant_records = True

        report = self.decision_engine.evaluate_arm(
            arm_id=arm_id,
            telemetry=telemetry,
            deterministic_result=safety_res,
            spoof_result=spoof_res,
            ml_prediction=ml_res,
            log_summary=log_summary
        )

        return report

    def format_report_card(self, report: ArmDiagnosticReport) -> str:
        t = report.telemetry_snapshot
        status = report.final_status.value

        symbols = {
            "SAFE": "[PASS] SAFE",
            "MONITOR": "[WARN] MONITOR",
            "SCHEDULE_MAINTENANCE": "[ALERT] SCHEDULE MAINTENANCE",
            "EMERGENCY_STOP": "[CRITICAL] EMERGENCY STOP"
        }
        status_banner = symbols.get(status, status)

        lines = [
            "=" * 78,
            f"  ROBOTIC ARM JOINT HEALTH REPORT: {report.arm_id}",
            "=" * 78,
            f"  FINAL SYSTEM DECISION:  >> {status_banner} <<",
            f"  PRIMARY TRIGGER:        {report.primary_reason}",
            f"  FUSED RISK SCORE:       {report.fused_risk_score:.3f} / 1.000",
            "-" * 78,
            "  TELEMETRY SNAPSHOT:",
            f"    * Shoulder: {t.get('shoulder_torque_nm', 0.0):.1f} Nm | {t.get('shoulder_temp_c', 0.0):.1f} °C | {t.get('shoulder_vib_mms', 0.0):.2f} mm/s",
            f"    * Elbow:    {t.get('elbow_torque_nm', 0.0):.1f} Nm | {t.get('elbow_temp_c', 0.0):.1f} °C | {t.get('elbow_vib_mms', 0.0):.2f} mm/s",
            f"    * Wrist:    {t.get('wrist_torque_nm', 0.0):.1f} Nm | {t.get('wrist_temp_c', 0.0):.1f} °C | {t.get('wrist_vib_mms', 0.0):.2f} mm/s",
            "-" * 78,
            "  SIGNAL BREAKDOWN:",
            f"  1. Deterministic Safety Guard: {'TRIPPED' if report.deterministic_result.is_tripped else 'CLEAR'}",
            f"     -> {report.deterministic_result.summary()}",
            f"  2. Spoofed-Sensor Defense:     {'SPOOFING DETECTED' if report.spoof_result.is_spoofed else 'AUTHENTIC'}",
            f"     -> {report.spoof_result.summary()}",
        ]

        if report.ml_prediction:
            ml = report.ml_prediction
            lines.extend([
                f"  3. Predictive Wear ML Model:   {ml.predicted_label} (Risk: {ml.wear_risk_score:.1%}, Most Worn: {ml.most_worn_joint.upper()})",
                f"     -> Probabilities: Healthy={ml.probabilities['HEALTHY']:.2f}, Degraded={ml.probabilities['DEGRADED']:.2f}, Critical={ml.probabilities['CRITICAL']:.2f}"
            ])
        else:
            lines.append("  3. Predictive Wear ML Model:   N/A (Bypassed due to emergency trip)")

        lines.extend([
            f"  4. Maintenance Log Analysis:   {report.log_summary.overall_severity} SEVERITY (Score: {report.log_summary.max_severity_score:.2f})",
            f"     -> Affected Joints: {', '.join([j.upper() for j in report.log_summary.affected_joints]) if report.log_summary.affected_joints else 'None'}",
            f"     -> Findings: {'; '.join(report.log_summary.summary_findings[:2])}",
            "-" * 78,
            f"  JUSTIFICATION: {report.detailed_justification}",
            f"  ACTION:        {report.recommended_action}",
            "=" * 78
        ])
        return "\n".join(lines)
