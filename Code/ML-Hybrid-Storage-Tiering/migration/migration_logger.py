"""
Migration Logger (Phase 13) for ML-Driven Hybrid Storage Staging.

Records all migration attempts, including successes, failures, and skips.
Maintains an auditable CSV log of all migration decisions and outcomes.
"""

from pathlib import Path
import pandas as pd
import uuid
import datetime
from typing import Dict, Any
import os

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


# Define the complete schema for the migration log
MIGRATION_LOG_COLUMNS = [
    "migration_id",
    "timestamp",
    "file_id",
    "source_tier",
    "destination_tier",
    "logical_file_size_bytes",
    "physical_simulation_size_bytes",
    "predicted_future_access_count",
    "predicted_hotspot",
    "heat_score",
    "migration_cost_sec",
    "expected_benefit_sec",
    "payback_ratio",
    "decision",
    "decision_reason",
    "migration_status",
    "actual_migration_duration_seconds",
    "checksum_verified",
    "error_message",
]


def _initialize_log_file(log_path: Path, columns: list):
    """Create the log file with headers if it doesn't exist."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        df_empty = pd.DataFrame(columns=columns)
        df_empty.to_csv(log_path, index=False)


def log_migration(
    decision_record: Dict[str, Any],
    migration_result: Dict[str, Any],
    logical_size_bytes: int,
) -> str:
    """
    Log a migration attempt to migration_log.csv.
    
    Args:
        decision_record: Dictionary containing ML prediction and decision info.
        migration_result: Dictionary returned by migrate_file().
        logical_size_bytes: Original file size from metadata.
        
    Returns:
        migration_id
    """
    _initialize_log_file(config.MIGRATION_LOG_PATH, MIGRATION_LOG_COLUMNS)
    
    # Generate unique ID (Timestamp + Short UUID)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    migration_id = f"MIG_{ts}_{short_uuid}"
    
    # Construct complete log record
    record = {
        "migration_id": migration_id,
        "timestamp": datetime.datetime.now().isoformat(),
        
        # Decision Info
        "file_id": decision_record.get("file_id", migration_result.get("file_id")),
        "predicted_future_access_count": decision_record.get("predicted_future_access_count", ""),
        "predicted_hotspot": decision_record.get("predicted_hotspot", ""),
        "heat_score": decision_record.get("heat_score", ""),
        "migration_cost_sec": decision_record.get("migration_cost_sec", ""),
        "expected_benefit_sec": decision_record.get("expected_benefit_sec", ""),
        "payback_ratio": decision_record.get("payback_ratio", ""),
        "decision": decision_record.get("decision", ""),
        "decision_reason": decision_record.get("decision_reason", ""),
        
        # Migration Result Info
        "source_tier": migration_result.get("source_tier", ""),
        "destination_tier": migration_result.get("destination_tier", ""),
        "logical_file_size_bytes": logical_size_bytes,
        "physical_simulation_size_bytes": migration_result.get("physical_size_bytes", ""),
        "migration_status": migration_result.get("status", ""),
        "actual_migration_duration_seconds": migration_result.get("duration_seconds", ""),
        "checksum_verified": migration_result.get("checksum_verified", ""),
        "error_message": migration_result.get("error_message", ""),
    }
    
    # Convert to DataFrame and append
    df_record = pd.DataFrame([record], columns=MIGRATION_LOG_COLUMNS)
    df_record.to_csv(config.MIGRATION_LOG_PATH, mode="a", header=False, index=False)
    
    return migration_id


if __name__ == "__main__":
    print("Testing Migration Logger...")
    
    test_decision = {
        "file_id": "file_test_log",
        "predicted_future_access_count": 15.5,
        "predicted_hotspot": 1,
        "heat_score": 0.85,
        "migration_cost_sec": 1.2,
        "expected_benefit_sec": 2.5,
        "payback_ratio": 2.08,
        "decision": "MIGRATE_TO_SSD",
        "decision_reason": "Test migration",
    }
    
    test_result = {
        "file_id": "file_test_log",
        "source_tier": 0,
        "destination_tier": 1,
        "status": "SUCCESS",
        "physical_size_bytes": 1048576,
        "duration_seconds": 0.05,
        "checksum_verified": True,
        "error_message": "",
    }
    
    mid = log_migration(test_decision, test_result, 50000000)
    print(f"Logged mock migration with ID: {mid}")
    print(f"Log file exists: {config.MIGRATION_LOG_PATH.exists()}")
