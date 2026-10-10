"""
DAY 1 · HANDS-ON — Build Your Rule Engine
23IF02 System Automation

Work through the TODOs in order. Each one matches a step on the
"Hands-on · Build Your Rule Engine" slide. Run this file after each
step to see your progress:

    python hands_on_starter.py

Need the answer? See hands_on_solution.py — but try it yourself first.
"""

import random
import time

LOG_FILE = "day1_log.txt"


# ---------------------------------------------------------------------------
# STEP 1 — Generate simulated sensor data
# ---------------------------------------------------------------------------
def generate_sensor_data():
    """Return a dict with at least 'temperature', 'pressure', 'vibration'."""
    # TODO: fill in realistic random values, e.g. temperature 50-95 (deg C)
    return {
        "temperature": round(random.uniform(50,95),2),   # TODO
        "pressure": round(random.uniform(1.0,10.0),2),      # TODO
        "vibration": round(random.uniform(0.1,5.0),2),     # TODO
        "current":round(random.uniform(5.0,20.0))
    }
    # STRETCH GOAL (after the 8 steps below all work): real sensors aren't
    # perfectly clean. Try making one field randomly return None (a
    # dropout) or spike far outside its normal range every so often, then
    # update evaluate_rules() so a dropout doesn't crash your engine.


# ---------------------------------------------------------------------------
# STEP 2 — Write your first automation rules
# STEP 7 — Add a rule of your own (add another dict to this list)
# STEP 8 — Add one priority rule (lower "priority" number checked first)
# ---------------------------------------------------------------------------
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
        "condition": lambda r: r.get("current", 0) > 25,
        "action": "REDUCE_LOAD",
    },
    {
        "name": "TEMP_WARNING",
        "priority": 5,
        "condition": lambda r: 70 <= r["temperature"] <= 80,
        "action": "ENABLE_COOLING",
    },
]

# ---------------------------------------------------------------------------
# STEP 3 — Detect machine state from the rules
# ---------------------------------------------------------------------------
def evaluate_rules(reading, rules):
    """Return (rule_name, action) for the first matching rule, checked in
    priority order (lowest number first). If nothing matches, return
    ("NO_RULE_FIRED", "CONTINUE_NORMAL")."""
    # TODO: sort `rules` by priority, then check each rule's condition(reading)
    rules=sorted(rules,key=lambda r:r["priority"])
    for rule in rules:
        try:
            if rule["condition"](reading):
                return rule["name"], rule["action"]
        except (KeyError, TypeError):
            continue
    return "NO_RULE_FIRED", "CONTINUE_NORMAL"



# ---------------------------------------------------------------------------
# STEP 4 — Trigger an action when a rule fires
# ---------------------------------------------------------------------------
def take_action(action):
    """Return a human-readable string describing what happens for this action."""
    # TODO: map action names (e.g. "STOP_MACHINE") to a message
    action_map = {
        "EMERGENCY_SHUTDOWN": "CUT POWER IMMEDIATELY: Dual critical threshold breached!",
        "STOP_MACHINE": "HALT: Machine stopped safely due to unsafe operating parameters.",
        "INSPECT_MACHINE": "ALERT: Schedule maintenance technician for inspection.",
        "REDUCE_LOAD": "THROTTLE: Lowering motor output to shed electrical load.",
        "ENABLE_COOLING": "MAINTAIN: Auxiliary coolant fans activated.",
        "CONTINUE_NORMAL": "NOMINAL: All parameters operating within normal envelope.",
    }
    return action_map.get(action, f"(no handler yet for action: {action})")


# ---------------------------------------------------------------------------
# STEP 5 — Log every decision made
# ---------------------------------------------------------------------------
def log_decision(reading, rule_name, action):
    """Append one line to LOG_FILE with the reading, rule fired, and action."""
    # TODO: open(LOG_FILE, "a") and write a line
    with open(LOG_FILE, "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] RULE: {rule_name} | ACTION: {action} | READINGS: {reading}\n")


# ---------------------------------------------------------------------------
# STEP 6 — Add a second sensor to the system
# TODO: add e.g. "current" to generate_sensor_data(), then use it in a rule
# ---------------------------------------------------------------------------


def main():
    for tick in range(1, 11):
        reading = generate_sensor_data()
        rule_name, action = evaluate_rules(reading, RULES)
        outcome = take_action(action)

        print(f"[Tick {tick:02d}] SENSE -> {reading}")
        print(f"           THINK -> {rule_name}")
        print(f"           ACT   -> {outcome}\n")

        log_decision(reading, rule_name, action)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
