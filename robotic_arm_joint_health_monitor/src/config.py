from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
HISTORICAL_DATA_FILE = DATA_DIR / "historical_joint_data.csv"
MAINTENANCE_LOG_FILE = DATA_DIR / "maintenance_logs.txt"


class ArmStatus(str, Enum):
    SAFE = "SAFE"
    MONITOR = "MONITOR"
    SCHEDULE_MAINTENANCE = "SCHEDULE_MAINTENANCE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class WearClass(int, Enum):
    HEALTHY = 0
    DEGRADED = 1
    CRITICAL = 2


@dataclass(frozen=True)
class JointThresholds:
    max_torque_nm: float       
    max_temp_c: float          
    max_vib_mms: float        

SAFETY_THRESHOLDS: Dict[str, JointThresholds] = {
    "shoulder": JointThresholds(
        max_torque_nm=175.0,  
        max_temp_c=85.0,       
        max_vib_mms=12.0       
    ),
    "elbow": JointThresholds(
        max_torque_nm=120.0,   
        max_temp_c=85.0,      
        max_vib_mms=11.0   
    ),
    "wrist": JointThresholds(
        max_torque_nm=60.0, 
        max_temp_c=80.0,       
        max_vib_mms=10.0       
    )
}


@dataclass(frozen=True)
class SpoofDefenseConfig:
    min_physical_torque_under_high_stress: Dict[str, float] = None
    high_vib_trigger_mms: float = 6.0     
    high_temp_trigger_c: float = 65.0      
    frozen_variance_epsilon: float = 1e-4
    kinematic_shoulder_min_ratio: float = 0.35

SPOOF_CONFIG = SpoofDefenseConfig(
    min_physical_torque_under_high_stress={
        "shoulder": 35.0,  
        "elbow": 25.0,    
        "wrist": 12.0      
    }
)
