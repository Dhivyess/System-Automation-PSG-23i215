import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
from src.config import MAINTENANCE_LOG_FILE


class LogSeverity(str):
    CLEAN = "CLEAN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class ParsedLogEntry:
    timestamp: Optional[str]
    arm_id: str
    technician: Optional[str]
    raw_text: str
    is_mechanically_relevant: bool
    severity: str
    severity_score: float             
    mentioned_joints: List[str]
    detected_symptoms: List[str]


@dataclass
class ArmLogSummary:
    arm_id: str
    overall_severity: str
    max_severity_score: float
    affected_joints: List[str]
    has_relevant_records: bool
    summary_findings: List[str]
    relevant_entries: List[ParsedLogEntry] = field(default_factory=list)


class MaintenanceLogParser:
    JOINT_PATTERNS = {
        "shoulder": [r"\bshoulder\b", r"\baxis\s*[12]\b", r"\bbase\s*joint\b", r"\bj1\b", r"\bj2\b"],
        "elbow": [r"\belbow\b", r"\baxis\s*3\b", r"\bj3\b", r"\belbow\s*gear\b"],
        "wrist": [r"\bwrist\b", r"\baxis\s*[456]\b", r"\bj[456]\b", r"\bflange\b", r"\bwrist\s*bearing\b"]
    }

    HIGH_SEVERITY_KEYWORDS = [
        "grinding", "metallic", "metal shavings", "lockup", "imminent",
        "severe shudder", "shuddering", "hot to the touch", "tooth fracture",
        "gear shearing", "excessive heat", "dark grease discoloration", "heavy play"
    ]

    MEDIUM_SEVERITY_KEYWORDS = [
        "backlash", "chatter", "play", "clicking", "clicking noise",
        "jerkiness", "vibration felt", "abnormal noise", "grease discoloration",
        "seepage", "oil leak", "intermittent chatter"
    ]

    LOW_SEVERITY_KEYWORDS = [
        "slight", "minor", "warm", "humming", "inspection due",
        "minor vibration", "minor play", "slight humming"
    ]

    NOMINAL_KEYWORDS = [
        "visual inspection", "routine", "normal", "within spec", "repeatability",
        "measured within", "calibration verification", "no abnormalities",
        "all axis seals intact", "nominally smooth", "logged as nominal", "clean"
    ]

    NOISE_PATTERNS = [
        r"coffee cup", r"shop towels?", r"swept", r"lunch", r"caution tape",
        r"yellow tape", r"zip tie", r"compressor hose", r"clipboard", r"binder"
    ]

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or MAINTENANCE_LOG_FILE
        self.entries: List[ParsedLogEntry] = []
        if self.log_path and self.log_path.exists():
            self.load_and_parse(self.log_path)

    def parse_entry(self, line: str) -> Optional[ParsedLogEntry]:
        clean_line = line.strip()
        if not clean_line or clean_line.startswith("#"):
            return None

        header_pattern = r"^\[(.*?)\]\s+(ARM-\d+|[A-Za-z0-9_-]+)\s+([A-Za-z0-9_-]+)?:\s*(.*)$"
        match = re.match(header_pattern, clean_line)

        if match:
            timestamp, arm_id, technician, text = match.groups()
        else:
            timestamp = None
            technician = None
            arm_match = re.search(r"\b(ARM-\d{1,2}|ARM\d{1,2})\b", clean_line, re.IGNORECASE)
            if arm_match:
                arm_id = arm_match.group(1).upper()
                text = clean_line
            else:
                arm_id = "UNKNOWN"
                text = clean_line

        text_lower = text.lower()
        is_pure_noise = any(re.search(pat, text_lower) for pat in self.NOISE_PATTERNS)
        has_mechanical_mention = any(
            any(re.search(p, text_lower) for p in pats)
            for pats in self.JOINT_PATTERNS.values()
        )
        detected_joints: List[str] = []
        for joint, patterns in self.JOINT_PATTERNS.items():
            if any(re.search(p, text_lower) for p in patterns):
                detected_joints.append(joint)
        detected_symptoms: List[str] = []
        severity = LogSeverity.CLEAN
        score = 0.0

        for kw in self.HIGH_SEVERITY_KEYWORDS:
            if kw in text_lower:
                detected_symptoms.append(kw)
                severity = LogSeverity.HIGH
                score = max(score, 0.90)

        if severity != LogSeverity.HIGH:
            for kw in self.MEDIUM_SEVERITY_KEYWORDS:
                if kw in text_lower:
                    detected_symptoms.append(kw)
                    severity = LogSeverity.MEDIUM
                    score = max(score, 0.60)

        if severity not in [LogSeverity.HIGH, LogSeverity.MEDIUM]:
            for kw in self.LOW_SEVERITY_KEYWORDS:
                if kw in text_lower:
                    detected_symptoms.append(kw)
                    severity = LogSeverity.LOW
                    score = max(score, 0.35)

        if severity == LogSeverity.CLEAN:
            for kw in self.NOMINAL_KEYWORDS:
                if kw in text_lower:
                    detected_symptoms.append(f"nominal: {kw}")
                    score = 0.0

        is_relevant = bool((detected_joints or detected_symptoms) and not (is_pure_noise and not has_mechanical_mention))

        if is_pure_noise and not (detected_joints or severity != LogSeverity.CLEAN):
            is_relevant = False
            score = 0.0
            severity = LogSeverity.CLEAN

        return ParsedLogEntry(
            timestamp=timestamp,
            arm_id=arm_id,
            technician=technician,
            raw_text=text,
            is_mechanically_relevant=is_relevant,
            severity=severity,
            severity_score=round(score, 2),
            mentioned_joints=detected_joints,
            detected_symptoms=detected_symptoms
        )

    def load_and_parse(self, file_path: Path):
        self.entries = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = self.parse_entry(line)
                if parsed:
                    self.entries.append(parsed)

    def get_arm_summary(self, arm_id: str) -> ArmLogSummary:
        arm_id_norm = arm_id.upper()
        matching = [e for e in self.entries if e.arm_id.upper() == arm_id_norm]

        if not matching:
            return ArmLogSummary(
                arm_id=arm_id_norm,
                overall_severity=LogSeverity.CLEAN,
                max_severity_score=0.0,
                affected_joints=[],
                has_relevant_records=False,
                summary_findings=["No prior technician logs recorded for this arm."]
            )

        relevant = [e for e in matching if e.is_mechanically_relevant]
        if not relevant:
            return ArmLogSummary(
                arm_id=arm_id_norm,
                overall_severity=LogSeverity.CLEAN,
                max_severity_score=0.0,
                affected_joints=[],
                has_relevant_records=False,
                summary_findings=["All records for this arm are non-mechanical routine notes (shift sweep, cleaning, coffee)."]
            )

        max_entry = max(relevant, key=lambda x: x.severity_score)
        all_joints: Set[str] = set()
        for e in relevant:
            all_joints.update(e.mentioned_joints)

        findings = []
        for e in relevant:
            if e.detected_symptoms:
                findings.append(f"{e.severity}: {', '.join(e.detected_symptoms)} ('{e.raw_text[:60]}...')")

        return ArmLogSummary(
            arm_id=arm_id_norm,
            overall_severity=max_entry.severity,
            max_severity_score=max_entry.severity_score,
            affected_joints=sorted(list(all_joints)),
            has_relevant_records=True,
            summary_findings=findings,
            relevant_entries=relevant
        )
