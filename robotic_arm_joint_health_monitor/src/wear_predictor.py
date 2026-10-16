import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.config import HISTORICAL_DATA_FILE, WearClass


@dataclass
class WearPredictionResult:
    predicted_class: int
    predicted_label: str
    wear_risk_score: float             
    probabilities: Dict[str, float]    
    most_worn_joint: str               
    top_contributing_features: List[Tuple[str, float]]
    confidence: float


class PredictiveWearModel:
    FEATURE_COLUMNS = [
        "shoulder_torque_nm", "shoulder_temp_c", "shoulder_vib_mms",
        "elbow_torque_nm", "elbow_temp_c", "elbow_vib_mms",
        "wrist_torque_nm", "wrist_temp_c", "wrist_vib_mms",
        "total_vib_power", "max_joint_vib", "total_thermal_stress",
        "shoulder_power_proxy", "elbow_power_proxy", "wrist_power_proxy"
    ]

    LABEL_MAP = {
        0: "HEALTHY",
        1: "DEGRADED",
        2: "CRITICAL"
    }

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or (HISTORICAL_DATA_FILE.parent / "wear_model.pkl")
        self.classifier: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained: bool = False
        self.training_metrics: Dict[str, Any] = {}

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = df.copy()
        feat["total_vib_power"] = (
            feat["shoulder_vib_mms"] ** 2 +
            feat["elbow_vib_mms"] ** 2 +
            feat["wrist_vib_mms"] ** 2
        )
        feat["max_joint_vib"] = feat[["shoulder_vib_mms", "elbow_vib_mms", "wrist_vib_mms"]].max(axis=1)
        feat["total_thermal_stress"] = (
            feat["shoulder_temp_c"] +
            feat["elbow_temp_c"] +
            feat["wrist_temp_c"]
        )
        feat["shoulder_power_proxy"] = feat["shoulder_torque_nm"] * feat["shoulder_vib_mms"]
        feat["elbow_power_proxy"] = feat["elbow_torque_nm"] * feat["elbow_vib_mms"]
        feat["wrist_power_proxy"] = feat["wrist_torque_nm"] * feat["wrist_vib_mms"]

        return feat[self.FEATURE_COLUMNS]

    def train(self, data_path: Optional[Path] = None) -> Dict[str, Any]:
        dataset_path = data_path or HISTORICAL_DATA_FILE
        if not dataset_path.exists():
            raise FileNotFoundError(f"Historical dataset not found at {dataset_path}")

        df = pd.read_csv(dataset_path)
        X = self.extract_features(df)
        y = df["wear_class"].values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42
        )

        cv_scores = cross_val_score(rf, X, y, cv=skf, scoring="f1_macro")
        rf.fit(X, y)
        y_pred = rf.predict(X)

        self.classifier = rf
        self.is_trained = True
        self.training_metrics = {
            "num_samples": len(df),
            "cv_macro_f1_mean": float(np.mean(cv_scores)),
            "cv_macro_f1_std": float(np.std(cv_scores)),
            "train_accuracy": float(accuracy_score(y, y_pred)),
            "feature_importances": dict(zip(self.FEATURE_COLUMNS, [float(v) for v in rf.feature_importances_]))
        }

        self.save()
        return self.training_metrics

    def predict(self, telemetry: Dict[str, float]) -> WearPredictionResult:
        if not self.is_trained or self.classifier is None:
            if self.model_path.exists():
                self.load()
            else:
                self.train()

        input_df = pd.DataFrame([telemetry])
        X = self.extract_features(input_df)

        pred_class = int(self.classifier.predict(X)[0])
        probas = self.classifier.predict_proba(X)[0]

        prob_dict = {
            "HEALTHY": float(probas[0]),
            "DEGRADED": float(probas[1]),
            "CRITICAL": float(probas[2])
        }
        wear_risk_score = float(0.0 * prob_dict["HEALTHY"] + 0.50 * prob_dict["DEGRADED"] + 1.00 * prob_dict["CRITICAL"])
        confidence = float(np.max(probas))

        sh_dev = (telemetry.get("shoulder_vib_mms", 1.5) / 1.6) + (telemetry.get("shoulder_temp_c", 40) / 42.0)
        el_dev = (telemetry.get("elbow_vib_mms", 1.4) / 1.4) + (telemetry.get("elbow_temp_c", 38) / 38.0)
        wr_dev = (telemetry.get("wrist_vib_mms", 1.2) / 1.2) + (telemetry.get("wrist_temp_c", 35) / 35.0)

        joint_devs = {"shoulder": sh_dev, "elbow": el_dev, "wrist": wr_dev}
        most_worn_joint = max(joint_devs, key=joint_devs.get)

        sorted_feats = sorted(
            self.training_metrics.get("feature_importances", {}).items(),
            key=lambda item: item[1],
            reverse=True
        )[:3]

        return WearPredictionResult(
            predicted_class=pred_class,
            predicted_label=self.LABEL_MAP[pred_class],
            wear_risk_score=round(wear_risk_score, 3),
            probabilities={k: round(v, 3) for k, v in prob_dict.items()},
            most_worn_joint=most_worn_joint,
            top_contributing_features=sorted_feats,
            confidence=round(confidence, 3)
        )

    def save(self):
        payload = {
            "classifier": self.classifier,
            "metrics": self.training_metrics
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(payload, f)

    def load(self):
        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)
            self.classifier = payload["classifier"]
            self.training_metrics = payload["metrics"]
            self.is_trained = True
