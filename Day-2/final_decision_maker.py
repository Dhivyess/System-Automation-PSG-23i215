import random
import time
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

LOG_FILE = r"C:\Users\PL\Documents\System-Automation-PSG-23i215\day1_log.txt"
DATASET_PATH = r"C:\Users\PL\Documents\System-Automation-PSG-23i215\Day-2\Data\day2_sample.csv"

def train_sensor_model(data_path: Path):

    df = pd.read_csv(data_path)
    features = ['temperature', 'pressure', 'vibration', 'rpm', 'current']
    X = df[features]
    y = df['state']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[MODEL] Trained RandomForestClassifier | Test Accuracy: {acc * 100:.2f}%\n")
    return model, features

def generate_sensor_data():
    return {
        "temperature": round(random.uniform(50, 95), 2),
        "pressure": round(random.uniform(1.0, 10.0), 2),
        "vibration": round(random.uniform(0.1, 5.0), 2),
        "rpm": random.randint(500, 3500),
        "current": round(random.uniform(5.0, 20.0), 2),
    }

RULES = [
    {
        "name": "CRITICAL_COMBO_OVERHEAT",
        "priority": 0,
        "condition": lambda r: r["temperature"] > 90 and r["pressure"] > 8,
        "action": "EMERGENCY_SHUTDOWN",
    },
    {
        "name": "HIGH_TEMP",
        "priority": 1,
        "condition": lambda r: r["temperature"] > 80,
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 2,
        "condition": lambda r: r["pressure"] > 4,
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 3,
        "condition": lambda r: r["vibration"] > 2.5,
        "action": "INSPECT_MACHINE",
    },
    {
        "name": "MOTOR_OVERCURRENT",
        "priority": 4,
        "condition": lambda r: r.get("current", 0) > 18.0,
        "action": "REDUCE_LOAD",
    },
    {
        "name": "TEMP_WARNING",
        "priority": 5,
        "condition": lambda r: 70 <= r["temperature"] <= 80,
        "action": "ENABLE_COOLING",
    },
]


def evaluate_rules(reading, rules):
    sorted_rules = sorted(rules, key=lambda r: r["priority"])
    for rule in sorted_rules:
        try:
            if rule["condition"](reading):
                return rule["name"], rule["action"]
        except (KeyError, TypeError):
            continue
    return "NO_RULE_FIRED", "CONTINUE_NORMAL"


def take_action(action):
    action_map = {
        "EMERGENCY_SHUTDOWN": "CUT POWER IMMEDIATELY: Dual critical threshold breached!",
        "STOP_MACHINE": "HALT: Machine stopped safely due to unsafe operating parameters.",
        "INSPECT_MACHINE": "ALERT: Schedule maintenance technician for inspection.",
        "REDUCE_LOAD": "THROTTLE: Lowering motor output to shed electrical load.",
        "ENABLE_COOLING": "MAINTAIN: Auxiliary coolant fans activated.",
        "CONTINUE_NORMAL": "NOMINAL: All parameters operating within normal envelope.",
    }
    return action_map.get(action, f"(no handler yet for action: {action})")


def log_decision(reading, rule_name, action, ml_pred):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"[{timestamp}] RULE: {rule_name} | ACTION: {action} | "
            f"ML_PRED: {ml_pred} | READINGS: {reading}\n"
        )


def main():
    model, feature_names = train_sensor_model(DATASET_PATH)

    for tick in range(1, 11):
        reading = generate_sensor_data()

        rule_name, action = evaluate_rules(reading, RULES)
        outcome = take_action(action)

        sample_df = pd.DataFrame([[reading[k] for k in feature_names]], columns=feature_names)
        ml_prediction = model.predict(sample_df)[0]

        print(f"[Tick {tick:02d}] SENSE      -> {reading}")
        print(f"          RULE_ACT   -> {rule_name} ({outcome})")
        print(f"          ML_PREDICT -> State classified as: '{ml_prediction.upper()}'\n")

        log_decision(reading, rule_name, action, ml_prediction)
        time.sleep(0.2)


if __name__ == "__main__":
    main()