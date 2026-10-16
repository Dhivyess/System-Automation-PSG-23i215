"""
Synthetic Dataset Generator for 6-Axis Robotic Arm Joint Health Monitoring.
Generates realistic multivariate sensor telemetry representing:
- Joint Torque (Nm)
- Joint Temperature (°C)
- Joint Vibration (mm/s RMS)
across Shoulder, Elbow, and Wrist joints under various health states.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_dataset(output_path: Path, n_samples: int = 150, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # Proportions: 50% Healthy, 30% Degraded (Incipient Wear), 20% Critical/High Wear
    n_healthy = int(n_samples * 0.50)
    n_degraded = int(n_samples * 0.30)
    n_critical = n_samples - n_healthy - n_degraded
    
    records = []
    
    # 1. Healthy / Normal Operations (Wear Class 0)
    for i in range(n_healthy):
        op_hours = np.random.uniform(50, 2000)
        cycles = int(op_hours * np.random.uniform(110, 130))
        
        # Nominal physical distributions with natural sensor noise
        sh_torque = np.random.normal(68.0, 7.5)     # Shoulder (Nm)
        sh_temp = np.random.normal(42.0, 3.5)       # Shoulder (°C)
        sh_vib = np.random.normal(1.6, 0.35)        # Shoulder (mm/s)
        
        el_torque = np.random.normal(42.0, 5.0)     # Elbow (Nm)
        el_temp = np.random.normal(38.0, 3.0)       # Elbow (°C)
        el_vib = np.random.normal(1.4, 0.30)        # Elbow (mm/s)
        
        wr_torque = np.random.normal(18.0, 2.5)     # Wrist (Nm)
        wr_temp = np.random.normal(35.0, 2.8)       # Wrist (°C)
        wr_vib = np.random.normal(1.2, 0.25)        # Wrist (mm/s)
        
        records.append({
            "sample_id": f"HIST-H{i+1:03d}",
            "arm_model": "KR-20-6AXIS",
            "operating_hours": round(op_hours, 1),
            "cycle_count": cycles,
            "shoulder_torque_nm": round(float(np.clip(sh_torque, 40.0, 95.0)), 2),
            "shoulder_temp_c": round(float(np.clip(sh_temp, 30.0, 55.0)), 2),
            "shoulder_vib_mms": round(float(np.clip(sh_vib, 0.6, 2.8)), 2),
            "elbow_torque_nm": round(float(np.clip(el_torque, 25.0, 60.0)), 2),
            "elbow_temp_c": round(float(np.clip(el_temp, 28.0, 50.0)), 2),
            "elbow_vib_mms": round(float(np.clip(el_vib, 0.5, 2.6)), 2),
            "wrist_torque_nm": round(float(np.clip(wr_torque, 10.0, 28.0)), 2),
            "wrist_temp_c": round(float(np.clip(wr_temp, 25.0, 48.0)), 2),
            "wrist_vib_mms": round(float(np.clip(wr_vib, 0.4, 2.4)), 2),
            "wear_class": 0,
            "wear_label": "HEALTHY",
            "failure_mode": "NONE"
        })
        
    # 2. Degraded / Incipient Wear Operations (Wear Class 1)
    # Joint wear shows elevated vibration harmonics, increased friction torque, and mild temperature creep
    wear_locations = ["SHOULDER", "ELBOW", "WRIST", "MULTI_JOINT"]
    degraded_modes = ["MICRO_PITTING", "LUBRICANT_DEGRADATION", "MINOR_BACKLASH"]
    
    for i in range(n_degraded):
        op_hours = np.random.uniform(2500, 7000)
        cycles = int(op_hours * np.random.uniform(115, 135))
        loc = np.random.choice(wear_locations)
        mode = np.random.choice(degraded_modes)
        
        # Base values elevated
        sh_torque = np.random.normal(82.0, 9.0)
        sh_temp = np.random.normal(52.0, 4.5)
        sh_vib = np.random.normal(3.8, 0.65)
        
        el_torque = np.random.normal(54.0, 6.5)
        el_temp = np.random.normal(48.0, 4.0)
        el_vib = np.random.normal(3.4, 0.60)
        
        wr_torque = np.random.normal(24.0, 3.5)
        wr_temp = np.random.normal(44.0, 3.8)
        wr_vib = np.random.normal(3.1, 0.55)
        
        # Spike the worn joint specifically
        if loc == "SHOULDER" or loc == "MULTI_JOINT":
            sh_torque += np.random.uniform(15.0, 30.0)
            sh_temp += np.random.uniform(8.0, 16.0)
            sh_vib += np.random.uniform(1.5, 3.0)
        if loc == "ELBOW" or loc == "MULTI_JOINT":
            el_torque += np.random.uniform(12.0, 24.0)
            el_temp += np.random.uniform(7.0, 14.0)
            el_vib += np.random.uniform(1.2, 2.8)
        if loc == "WRIST" or loc == "MULTI_JOINT":
            wr_torque += np.random.uniform(6.0, 14.0)
            wr_temp += np.random.uniform(6.0, 13.0)
            wr_vib += np.random.uniform(1.0, 2.5)
            
        records.append({
            "sample_id": f"HIST-D{i+1:03d}",
            "arm_model": "KR-20-6AXIS",
            "operating_hours": round(op_hours, 1),
            "cycle_count": cycles,
            "shoulder_torque_nm": round(float(np.clip(sh_torque, 55.0, 140.0)), 2),
            "shoulder_temp_c": round(float(np.clip(sh_temp, 40.0, 72.0)), 2),
            "shoulder_vib_mms": round(float(np.clip(sh_vib, 2.8, 6.8)), 2),
            "elbow_torque_nm": round(float(np.clip(el_torque, 35.0, 95.0)), 2),
            "elbow_temp_c": round(float(np.clip(el_temp, 35.0, 68.0)), 2),
            "elbow_vib_mms": round(float(np.clip(el_vib, 2.5, 6.2)), 2),
            "wrist_torque_nm": round(float(np.clip(wr_torque, 16.0, 42.0)), 2),
            "wrist_temp_c": round(float(np.clip(wr_temp, 32.0, 65.0)), 2),
            "wrist_vib_mms": round(float(np.clip(wr_vib, 2.2, 5.8)), 2),
            "wear_class": 1,
            "wear_label": "DEGRADED",
            "failure_mode": mode
        })
        
    # 3. Critical Wear / Severe Damage Operations (Wear Class 2)
    # Severe bearing flaking, harmonic drive tooth breakdown, severe friction
    critical_modes = ["BEARING_SPALLING", "HARMONIC_DRIVE_TOOTH_FRACTURE", "SEVERE_LUBE_STARVATION"]
    for i in range(n_critical):
        op_hours = np.random.uniform(7500, 15000)
        cycles = int(op_hours * np.random.uniform(120, 140))
        loc = np.random.choice(wear_locations)
        mode = np.random.choice(critical_modes)
        
        sh_torque = np.random.normal(115.0, 14.0)
        sh_temp = np.random.normal(68.0, 6.0)
        sh_vib = np.random.normal(6.5, 1.2)
        
        el_torque = np.random.normal(78.0, 10.0)
        el_temp = np.random.normal(64.0, 5.5)
        el_vib = np.random.normal(5.8, 1.1)
        
        wr_torque = np.random.normal(38.0, 6.0)
        wr_temp = np.random.normal(60.0, 5.0)
        wr_vib = np.random.normal(5.2, 1.0)
        
        # Further amplify the critical joint (just below or at emergency border)
        if loc in ["SHOULDER", "MULTI_JOINT"]:
            sh_torque += np.random.uniform(25.0, 50.0)
            sh_temp += np.random.uniform(10.0, 15.0)
            sh_vib += np.random.uniform(2.5, 4.5)
        if loc in ["ELBOW", "MULTI_JOINT"]:
            el_torque += np.random.uniform(20.0, 38.0)
            el_temp += np.random.uniform(9.0, 16.0)
            el_vib += np.random.uniform(2.2, 4.2)
        if loc in ["WRIST", "MULTI_JOINT"]:
            wr_torque += np.random.uniform(12.0, 20.0)
            wr_temp += np.random.uniform(8.0, 16.0)
            wr_vib += np.random.uniform(2.0, 4.0)
            
        records.append({
            "sample_id": f"HIST-C{i+1:03d}",
            "arm_model": "KR-20-6AXIS",
            "operating_hours": round(op_hours, 1),
            "cycle_count": cycles,
            "shoulder_torque_nm": round(float(np.clip(sh_torque, 75.0, 172.0)), 2),
            "shoulder_temp_c": round(float(np.clip(sh_temp, 55.0, 84.0)), 2),
            "shoulder_vib_mms": round(float(np.clip(sh_vib, 4.5, 11.5)), 2),
            "elbow_torque_nm": round(float(np.clip(el_torque, 50.0, 118.0)), 2),
            "elbow_temp_c": round(float(np.clip(el_temp, 50.0, 83.0)), 2),
            "elbow_vib_mms": round(float(np.clip(el_vib, 4.0, 10.5)), 2),
            "wrist_torque_nm": round(float(np.clip(wr_torque, 25.0, 58.0)), 2),
            "wrist_temp_c": round(float(np.clip(wr_temp, 45.0, 78.0)), 2),
            "wrist_vib_mms": round(float(np.clip(wr_vib, 3.8, 9.8)), 2),
            "wear_class": 2,
            "wear_label": "CRITICAL",
            "failure_mode": mode
        })
        
    df = pd.DataFrame(records)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} historical samples saved to {output_path}")
    print(df["wear_label"].value_counts())
    return df

if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent / "historical_joint_data.csv"
    generate_synthetic_dataset(out_file, n_samples=160, seed=42)
