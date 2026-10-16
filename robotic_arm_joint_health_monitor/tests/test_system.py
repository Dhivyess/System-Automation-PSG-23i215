import pytest
import pandas as pd
from pathlib import Path

from src.config import ArmStatus, SAFETY_THRESHOLDS, HISTORICAL_DATA_FILE, MAINTENANCE_LOG_FILE
from src.deterministic_safety import DeterministicSafetyGuard
from src.wear_predictor import PredictiveWearModel
from src.log_parser import MaintenanceLogParser, LogSeverity
from src.spoof_detector import SpoofDefenseValidator
from src.pipeline import RoboticArmMonitoringPipeline

def test_deterministic_safety_nominal_passes():
    guard = DeterministicSafetyGuard()
    nominal_telemetry = {
        "shoulder_torque_nm": 70.0, "shoulder_temp_c": 45.0, "shoulder_vib_mms": 1.8,
        "elbow_torque_nm": 45.0, "elbow_temp_c": 40.0, "elbow_vib_mms": 1.5,
        "wrist_torque_nm": 20.0, "wrist_temp_c": 35.0, "wrist_vib_mms": 1.2
    }
    res = guard.evaluate(nominal_telemetry)
    assert not res.is_tripped
    assert res.status == ArmStatus.SAFE
    assert res.violation_count == 0


def test_deterministic_safety_trips_on_single_metric():
    guard = DeterministicSafetyGuard()
    telemetry = {
        "shoulder_torque_nm": 178.5, "shoulder_temp_c": 45.0, "shoulder_vib_mms": 1.8,
        "elbow_torque_nm": 45.0, "elbow_temp_c": 40.0, "elbow_vib_mms": 1.5,
        "wrist_torque_nm": 20.0, "wrist_temp_c": 35.0, "wrist_vib_mms": 1.2
    }
    res = guard.evaluate(telemetry)
    assert res.is_tripped
    assert res.status == ArmStatus.EMERGENCY_STOP
    assert res.violation_count == 1
    assert res.violations[0].joint == "shoulder"
    assert res.violations[0].metric == "torque"


def test_deterministic_safety_trips_on_wrist_vibration():
    guard = DeterministicSafetyGuard()
    telemetry = {
        "shoulder_torque_nm": 70.0, "shoulder_temp_c": 45.0, "shoulder_vib_mms": 1.8,
        "elbow_torque_nm": 45.0, "elbow_temp_c": 40.0, "elbow_vib_mms": 1.5,
        "wrist_torque_nm": 20.0, "wrist_temp_c": 35.0, "wrist_vib_mms": 10.4
    }
    res = guard.evaluate(telemetry)
    assert res.is_tripped
    assert res.status == ArmStatus.EMERGENCY_STOP
    assert any(v.joint == "wrist" and v.metric == "vibration" for v in res.violations)


def test_deterministic_safety_zero_ml_dependency():
    """Verify that deterministic safety executes in under 1 millisecond and has no ML objects."""
    guard = DeterministicSafetyGuard()
    assert not hasattr(guard, "model")
    assert not hasattr(guard, "classifier")
    telemetry = {"shoulder_torque_nm": 200.0}
    res = guard.evaluate(telemetry)
    assert res.is_tripped
    assert res.execution_time_us < 1000.0


def test_historical_dataset_meets_deliverable_size():
    assert HISTORICAL_DATA_FILE.exists(), "Dataset file must exist"
    df = pd.read_csv(HISTORICAL_DATA_FILE)
    assert len(df) >= 50, f"Expected at least 50 historical samples, found {len(df)}"
    assert "wear_class" in df.columns
    assert "wear_label" in df.columns


def test_wear_model_training_and_predictions():
    model = PredictiveWearModel()
    metrics = model.train()
    assert model.is_trained
    assert metrics["cv_macro_f1_mean"] > 0.85, "Cross-validation macro F1 must be high"

    healthy_telemetry = {
        "shoulder_torque_nm": 65.0, "shoulder_temp_c": 40.0, "shoulder_vib_mms": 1.5,
        "elbow_torque_nm": 40.0, "elbow_temp_c": 36.0, "elbow_vib_mms": 1.3,
        "wrist_torque_nm": 18.0, "wrist_temp_c": 34.0, "wrist_vib_mms": 1.1
    }
    res_healthy = model.predict(healthy_telemetry)
    assert res_healthy.predicted_label == "HEALTHY"
    assert res_healthy.wear_risk_score < 0.35

    critical_telemetry = {
        "shoulder_torque_nm": 145.0, "shoulder_temp_c": 74.0, "shoulder_vib_mms": 8.8,
        "elbow_torque_nm": 85.0, "elbow_temp_c": 70.0, "elbow_vib_mms": 7.5,
        "wrist_torque_nm": 42.0, "wrist_temp_c": 62.0, "wrist_vib_mms": 6.8
    }
    res_critical = model.predict(critical_telemetry)
    assert res_critical.predicted_label in ["DEGRADED", "CRITICAL"]
    assert res_critical.wear_risk_score > 0.50

def test_maintenance_log_file_meets_deliverable_count():
    assert MAINTENANCE_LOG_FILE.exists(), "Maintenance log file must exist"
    parser = MaintenanceLogParser(MAINTENANCE_LOG_FILE)
    assert len(parser.entries) >= 8, f"Expected at least 8 entries, got {len(parser.entries)}"


def test_maintenance_log_noise_rejection():
    parser = MaintenanceLogParser()
    noisy_line = "[2026-10-10 07:15:22] ARM-01 Tech_Rajesh: Left a fresh roll of shop towels in cabinet B. Robot cell swept."
    parsed = parser.parse_entry(noisy_line)
    assert parsed is not None
    assert not parsed.is_mechanically_relevant
    assert parsed.severity == LogSeverity.CLEAN
    assert parsed.severity_score == 0.0


def test_maintenance_log_mechanical_extraction():
    parser = MaintenanceLogParser()
    critical_line = "[2026-10-15 08:45:10] ARM-08 Tech_Elena: Severe mechanical shudder detected during shoulder elevation. Metal shavings observed in grease sample."
    parsed = parser.parse_entry(critical_line)
    assert parsed is not None
    assert parsed.is_mechanically_relevant
    assert parsed.severity == LogSeverity.HIGH
    assert "shoulder" in parsed.mentioned_joints
    assert parsed.severity_score >= 0.85

def test_spoof_detection_torque_masking_attack():
    validator = SpoofDefenseValidator()
    tampered_telemetry = {
        "shoulder_torque_nm": 70.0, "shoulder_temp_c": 45.0, "shoulder_vib_mms": 2.0,
        "elbow_torque_nm": 45.0, "elbow_temp_c": 40.0, "elbow_vib_mms": 2.0,
        "wrist_torque_nm": 10.0,      
        "wrist_temp_c": 75.0,          
        "wrist_vib_mms": 8.5          
    }
    res = validator.validate(tampered_telemetry)
    assert res.is_spoofed, "Should catch torque-masking attack violating thermodynamic coupling"
    assert "wrist" in res.flagged_joints
    assert "TORQUE_MASKING_THERMODYNAMIC_VIOLATION" in res.attack_type


def test_spoof_detection_kinematic_inconsistency():
    validator = SpoofDefenseValidator()
    kinematic_anomaly = {
        "shoulder_torque_nm": 5.0,     
        "shoulder_temp_c": 42.0, "shoulder_vib_mms": 1.5,
        "elbow_torque_nm": 75.0, "elbow_temp_c": 50.0, "elbow_vib_mms": 3.0,
        "wrist_torque_nm": 35.0, "wrist_temp_c": 45.0, "wrist_vib_mms": 2.5
    }
    res = validator.validate(kinematic_anomaly)
    assert res.is_spoofed
    assert "shoulder" in res.flagged_joints
    assert "KINEMATIC_CHAIN_REACTION_VIOLATION" in res.attack_type


def test_spoof_detection_frozen_sensor_variance():
    validator = SpoofDefenseValidator()
    frozen_frame = {
        "shoulder_torque_nm": 65.0, "shoulder_temp_c": 42.0, "shoulder_vib_mms": 1.5,
        "elbow_torque_nm": 40.0, "elbow_temp_c": 38.0, "elbow_vib_mms": 1.4,
        "wrist_torque_nm": 18.0, "wrist_temp_c": 35.0, "wrist_vib_mms": 1.2
    }
    history = [frozen_frame.copy() for _ in range(6)]
    res = validator.validate(frozen_frame, history_window=history)
    assert res.is_spoofed
    assert "STATIC_REPLAY_FROZEN_TORQUE" in res.attack_type


def test_spoof_validator_passes_nominal_telemetry():
    validator = SpoofDefenseValidator()
    nominal = {
        "shoulder_torque_nm": 68.0, "shoulder_temp_c": 42.0, "shoulder_vib_mms": 1.6,
        "elbow_torque_nm": 42.0, "elbow_temp_c": 38.0, "elbow_vib_mms": 1.4,
        "wrist_torque_nm": 18.0, "wrist_temp_c": 35.0, "wrist_vib_mms": 1.2
    }
    res = validator.validate(nominal)
    assert not res.is_spoofed

def test_all_four_output_states_are_reachable():
    pipeline = RoboticArmMonitoringPipeline(auto_train=True)

    safe_telemetry = {
        "shoulder_torque_nm": 65.0, "shoulder_temp_c": 41.0, "shoulder_vib_mms": 1.4,
        "elbow_torque_nm": 39.0, "elbow_temp_c": 36.0, "elbow_vib_mms": 1.2,
        "wrist_torque_nm": 17.0, "wrist_temp_c": 33.0, "wrist_vib_mms": 1.0
    }
    rep_safe = pipeline.evaluate_arm("ARM-01", safe_telemetry)
    assert rep_safe.final_status == ArmStatus.SAFE

    monitor_telemetry = {
        "shoulder_torque_nm": 82.0, "shoulder_temp_c": 51.0, "shoulder_vib_mms": 3.6,
        "elbow_torque_nm": 52.0, "elbow_temp_c": 46.0, "elbow_vib_mms": 3.2,
        "wrist_torque_nm": 24.5, "wrist_temp_c": 44.0, "wrist_vib_mms": 3.4
    }
    rep_monitor = pipeline.evaluate_arm("ARM-02", monitor_telemetry)
    assert rep_monitor.final_status == ArmStatus.MONITOR

    maint_telemetry = {
        "shoulder_torque_nm": 98.0, "shoulder_temp_c": 58.0, "shoulder_vib_mms": 4.8,
        "elbow_torque_nm": 84.0, "elbow_temp_c": 64.0, "elbow_vib_mms": 5.9,
        "wrist_torque_nm": 29.0, "wrist_temp_c": 48.0, "wrist_vib_mms": 3.8
    }
    rep_maint = pipeline.evaluate_arm("ARM-03", maint_telemetry)
    assert rep_maint.final_status == ArmStatus.SCHEDULE_MAINTENANCE

    estop_telemetry = {
        "shoulder_torque_nm": 185.0, 
        "shoulder_temp_c": 88.0,
        "shoulder_vib_mms": 13.0,
        "elbow_torque_nm": 95.0, "elbow_temp_c": 68.0, "elbow_vib_mms": 6.5,
        "wrist_torque_nm": 32.0, "wrist_temp_c": 54.0, "wrist_vib_mms": 4.1
    }
    rep_estop = pipeline.evaluate_arm("ARM-08", estop_telemetry)
    assert rep_estop.final_status == ArmStatus.EMERGENCY_STOP
