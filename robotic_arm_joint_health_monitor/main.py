import sys
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import RoboticArmMonitoringPipeline
from src.config import ArmStatus


def run_demonstration():
    print("\n" + "#" * 80)
    print("  ROBOTIC ARM JOINT HEALTH MONITORING SYSTEM")
    print("#" * 80 + "\n")

    pipeline = RoboticArmMonitoringPipeline(auto_train=True)
    print(f"[OK] Pipeline initialized.")
    if pipeline.wear_model.is_trained:
        metrics = pipeline.wear_model.training_metrics
        print(f"[OK] Machine Learning Wear Model Active:")
        print(f"     * Training Samples: {metrics.get('num_samples', 'N/A')}")
        print(f"     * 5-Fold Stratified Macro-F1: {metrics.get('cv_macro_f1_mean', 0.0):.3f} (+/- {metrics.get('cv_macro_f1_std', 0.0):.3f})")
        print(f"     * Train Accuracy: {metrics.get('train_accuracy', 0.0):.3f}")

    print("\n" + "=" * 80)
    print("  DEMONSTRATION OF ALL 4 FINAL ARM STATES + SPOOF DEFENSE")
    print("=" * 80)

    arm_01_telemetry = {
        "shoulder_torque_nm": 65.4, "shoulder_temp_c": 41.2, "shoulder_vib_mms": 1.45,
        "elbow_torque_nm": 39.8, "elbow_temp_c": 36.5, "elbow_vib_mms": 1.30,
        "wrist_torque_nm": 17.2, "wrist_temp_c": 33.8, "wrist_vib_mms": 1.10
    }
    report_safe = pipeline.evaluate_arm(
        arm_id="ARM-01",
        telemetry=arm_01_telemetry
    )
    print("\n>>> DEMONSTRATION 1: EXPECTED STATUS -> SAFE")
    print(pipeline.format_report_card(report_safe))

    arm_02_telemetry = {
        "shoulder_torque_nm": 82.0, "shoulder_temp_c": 51.5, "shoulder_vib_mms": 3.60,
        "elbow_torque_nm": 52.0, "elbow_temp_c": 46.0, "elbow_vib_mms": 3.20,
        "wrist_torque_nm": 24.5, "wrist_temp_c": 44.0, "wrist_vib_mms": 3.45  
    }
    report_monitor = pipeline.evaluate_arm(
        arm_id="ARM-02",
        telemetry=arm_02_telemetry
    )
    print("\n>>> DEMONSTRATION 2: EXPECTED STATUS -> MONITOR")
    print(pipeline.format_report_card(report_monitor))

    arm_03_telemetry = {
        "shoulder_torque_nm": 98.0, "shoulder_temp_c": 58.0, "shoulder_vib_mms": 4.80,
        "elbow_torque_nm": 84.5, "elbow_temp_c": 64.0, "elbow_vib_mms": 5.90,  
        "wrist_torque_nm": 29.0, "wrist_temp_c": 48.0, "wrist_vib_mms": 3.80
    }
    report_maintenance = pipeline.evaluate_arm(
        arm_id="ARM-03",
        telemetry=arm_03_telemetry
    )
    print("\n>>> DEMONSTRATION 3: EXPECTED STATUS -> SCHEDULE_MAINTENANCE")
    print(pipeline.format_report_card(report_maintenance))

    arm_08_telemetry = {
        "shoulder_torque_nm": 182.0,  
        "shoulder_temp_c": 87.5,      
        "shoulder_vib_mms": 13.20,     
        "elbow_torque_nm": 95.0, "elbow_temp_c": 68.0, "elbow_vib_mms": 6.50,
        "wrist_torque_nm": 32.0, "wrist_temp_c": 54.0, "wrist_vib_mms": 4.10
    }
    report_estop_hard = pipeline.evaluate_arm(
        arm_id="ARM-08",
        telemetry=arm_08_telemetry
    )
    print("\n>>> DEMONSTRATION 4A: EXPECTED STATUS -> EMERGENCY_STOP (Deterministic Safety Rule Breach)")
    print(pipeline.format_report_card(report_estop_hard))

    arm_spoof_telemetry = {
        "shoulder_torque_nm": 72.0, "shoulder_temp_c": 45.0, "shoulder_vib_mms": 2.20,
        "elbow_torque_nm": 45.0, "elbow_temp_c": 42.0, "elbow_vib_mms": 2.10,
        "wrist_torque_nm": 10.5,   
        "wrist_temp_c": 76.5,         
        "wrist_vib_mms": 8.80        
    }
    report_estop_spoof = pipeline.evaluate_arm(
        arm_id="ARM-09",
        telemetry=arm_spoof_telemetry
    )
    print("\n>>> DEMONSTRATION 4B: EXPECTED STATUS -> EMERGENCY_STOP (Cyber-Physical Spoofed Sensor Defense)")
    print(pipeline.format_report_card(report_estop_spoof))

    print("\n" + "=" * 80)
    print("  EXECUTIVE SUMMARY TABLE: DEMONSTRATION RUNS")
    print("=" * 80)
    headers = f"{'ARM ID':<10} | {'FINAL STATUS':<22} | {'SAFETY':<8} | {'SPOOF DEF':<12} | {'ML WEAR':<10} | {'LOG SEVERITY':<12}"
    print(headers)
    print("-" * 80)

    demo_reports = [report_safe, report_monitor, report_maintenance, report_estop_hard, report_estop_spoof]
    for r in demo_reports:
        safety_txt = "TRIP" if r.deterministic_result.is_tripped else "PASS"
        spoof_txt = "ATTACK" if r.spoof_result.is_spoofed else "CLEAR"
        ml_txt = r.ml_prediction.predicted_label if r.ml_prediction else "N/A"
        log_txt = r.log_summary.overall_severity
        print(f"{r.arm_id:<10} | {r.final_status.value:<22} | {safety_txt:<8} | {spoof_txt:<12} | {ml_txt:<10} | {log_txt:<12}")
    print("=" * 80)

    statuses_reached = {r.final_status for r in demo_reports}
    all_four_reached = {ArmStatus.SAFE, ArmStatus.MONITOR, ArmStatus.SCHEDULE_MAINTENANCE, ArmStatus.EMERGENCY_STOP}.issubset(statuses_reached)

    print(f"\n[VERIFICATION] All 4 required statuses reachable and demonstrated: {all_four_reached}")
    print(f"  Reached states: {[s.value for s in statuses_reached]}\n")

    return demo_reports


if __name__ == "__main__":
    run_demonstration()
