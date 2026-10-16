# Robotic Arm Joint Health Monitoring System
**Course:** 23IF02 — System Automation (Bonus Challenge)

An end-to-end multi-modal cyber-physical health monitoring and diagnostic system for a fleet of 6-axis industrial assembly-line robotic arms. Evaluates real-time sensor telemetry (**joint torque**, **joint temperature**, and **vibration** across shoulder, elbow, and wrist joints) alongside unstructured technician maintenance logs.

---

## System Architecture & 5 Core Requirements

1. **Deterministic Immediate Safety Stop:**
   Instant, unambiguous emergency stop (`EMERGENCY_STOP`) triggered by deterministic physics rules if any single joint reading crosses a hazardous threshold. Strictly independent of the ML model to ensure failsafe execution under ISO 13849 PL-e standards.

2. **Predictive Wear Detection:**
   Supervised machine learning model (Random Forest / Ensemble Classifier) trained on a synthetic historical dataset of joint failure modes (gear micro-pitting, bearing spalling, lubricant starvation) to catch degradation before hardware limits are breached.

3. **Free-Text Maintenance Log Integration:**
   Natural language processing (NLP) parser that extracts actionable mechanical symptoms and affected joint entities from unstructured technician inspection notes while rejecting shop-floor operational noise (cleaning, coffee cups, shift handoffs).

4. **Spoofed-Sensor Cyber-Physical Defense:**
   Security validation layer that detects torque-masking attacks where fake "nominal" torque values are injected to mask physical joint binding. Validates thermodynamic energy dissipation ($\Delta T$ and vibration vs. torque drawn) and serial kinematic reaction force balance.

5. **Unified Per-Arm Decision Output:**
   Hierarchical arbitration engine fusing all four signals into one of four reachable, deterministic states:
   - `SAFE`
   - `MONITOR`
   - `SCHEDULE_MAINTENANCE`
   - `EMERGENCY_STOP`

---

## Project Structure

```
robotic_arm_joint_health_monitor/
├── data/
│   ├── generate_dataset.py          # Synthetic dataset generator script
│   ├── historical_joint_data.csv    # 160 labeled historical joint failure records
│   └── maintenance_logs.txt         # 13 unstructured free-text technician logs
├── src/
│   ├── __init__.py
│   ├── config.py                    # Physical bounds, ISO safety limits & enums
│   ├── deterministic_safety.py      # Hard rule-based emergency stop (Zero ML)
│   ├── wear_predictor.py            # Supervised ML predictive maintenance engine
│   ├── log_parser.py                # NLP parser for free-text technician notes
│   ├── spoof_detector.py            # Cyber-physical anti-spoofing validator
│   ├── decision_engine.py           # Multi-modal signal fusion & arbitration
│   └── pipeline.py                  # End-to-end orchestrator & report card generator
├── tests/
│   └── test_system.py               # Comprehensive pytest suite (14 unit/integration tests)
├── docs/
│   └── EXPLANATION.md               # Half-page technical write-up detailing all justifications
├── main.py                          # Demonstration runner executing all 4 states + spoof defense
├── pytest.ini                       # Test configuration
├── EXPLANATION.md                   # Half-page technical write-up (root copy)
└── README.md                        # Project documentation
```

---

## Quick Start & Execution

### 1. Run the Full Demonstration Pipeline
To run the automated demonstration displaying all 4 reached states, detailed report cards, and the spoof attack detection:
```bash
python main.py
```

### 2. Run the Automated Test Suite
To verify all 5 requirements with 100% passing tests:
```bash
pytest -v
```

### 3. Re-generate Historical Training Data
If you wish to re-generate the synthetic dataset:
```bash
python data/generate_dataset.py
```

---

## Technical Write-Up
For full justifications of safety thresholds, fusion decision logic, and spoof attack defenses/blind spots, please refer to [EXPLANATION.md]
