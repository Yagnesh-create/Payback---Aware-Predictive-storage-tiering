"""
Actual File Migration Engine (Phase 12) for ML-Driven Hybrid Storage Staging.

Performs simulated local file operations between storage/HDD/ and storage/SSD/.
Implements safe copy-verify-delete workflows with checksum verification.
"""

from pathlib import Path
import shutil
import hashlib
import time
import os
from typing import Dict, Any

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from migration.storage_manager import get_safe_file_path, get_file_tier


# Migration Status Constants
STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"
STATUS_SKIPPED = "SKIPPED"
STATUS_ALREADY_ON_DESTINATION = "ALREADY_ON_DESTINATION"


def _calculate_checksum(filepath: Path) -> str:
    """Calculate SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        return f"ERROR_{str(e)}"


def migrate_file(file_id: str, source_tier: int, dest_tier: int) -> Dict[str, Any]:
    """
    Safely migrate a file from source_tier to dest_tier.
    Workflow: COPY -> VERIFY CHECKSUM -> DELETE SOURCE.
    
    Returns a dictionary with migration results.
    """
    result = {
        "file_id": file_id,
        "source_tier": source_tier,
        "destination_tier": dest_tier,
        "status": STATUS_FAILED,
        "physical_size_bytes": 0,
        "duration_seconds": 0.0,
        "checksum_verified": False,
        "error_message": "",
    }
    
    # 1. Validations
    if source_tier == dest_tier:
        result["status"] = STATUS_ALREADY_ON_DESTINATION
        result["error_message"] = "Source and destination tiers are the same."
        return result
        
    try:
        source_path = get_safe_file_path(file_id, source_tier)
        dest_path = get_safe_file_path(file_id, dest_tier)
    except ValueError as e:
        result["error_message"] = f"Path generation error: {e}"
        return result

    # Check actual current tier (safeguard)
    actual_tier = get_file_tier(file_id)
    if actual_tier == dest_tier:
        result["status"] = STATUS_ALREADY_ON_DESTINATION
        result["error_message"] = "File is already on destination tier."
        return result
    if actual_tier != source_tier:
        result["error_message"] = f"File is not in expected source tier {source_tier}."
        return result
        
    if not source_path.exists():
        result["error_message"] = "Source file does not exist."
        return result
        
    if dest_path.exists():
        result["error_message"] = "Destination file already exists."
        return result

    # 2. Migration Execution
    try:
        physical_size = source_path.stat().st_size
        result["physical_size_bytes"] = physical_size
        
        # Calculate source checksum
        source_checksum = _calculate_checksum(source_path)
        if source_checksum.startswith("ERROR"):
            result["error_message"] = f"Source checksum failed: {source_checksum}"
            return result
            
        start_time = time.time()
        
        # COPY
        shutil.copy2(source_path, dest_path)
        
        # VERIFY
        dest_checksum = _calculate_checksum(dest_path)
        if dest_checksum != source_checksum:
            # Revert copy
            if dest_path.exists():
                os.remove(dest_path)
            result["error_message"] = "Checksum mismatch. Destination file deleted. Source preserved."
            return result
            
        result["checksum_verified"] = True
        
        # DELETE SOURCE
        os.remove(source_path)
        
        end_time = time.time()
        result["duration_seconds"] = end_time - start_time
        result["status"] = STATUS_SUCCESS
        
    except PermissionError as e:
        result["error_message"] = f"Permission denied: {e}"
    except OSError as e:
        result["error_message"] = f"OS/Disk error during migration: {e}"
    except Exception as e:
        result["error_message"] = f"Unexpected error: {e}"
        
    # Cleanup in case of failure during copy
    if result["status"] == STATUS_FAILED and dest_path.exists() and not result["checksum_verified"]:
        try:
            os.remove(dest_path)
        except:
            pass

    return result


if __name__ == "__main__":
    print("Testing Migration Manager...")
    from migration.storage_manager import initialize_storage_directories, get_safe_file_path
    
    initialize_storage_directories()
    
    # Create a dummy file in HDD
    test_file_id = "test_migration_01"
    hdd_path = get_safe_file_path(test_file_id, config.TIER_HDD_ENCODING)
    
    with open(hdd_path, "wb") as f:
        f.write(os.urandom(1024 * 512))  # 512 KB
        
    print(f"Created test file on HDD: {hdd_path.exists()}")
    
    # Migrate HDD -> SSD
    print("\nAttempting migration HDD -> SSD...")
    res1 = migrate_file(test_file_id, config.TIER_HDD_ENCODING, config.TIER_SSD_ENCODING)
    print(res1)
    
    # Migrate SSD -> HDD
    print("\nAttempting migration SSD -> HDD...")
    res2 = migrate_file(test_file_id, config.TIER_SSD_ENCODING, config.TIER_HDD_ENCODING)
    print(res2)
    
    # Clean up
    if hdd_path.exists():
        os.remove(hdd_path)
    ssd_path = get_safe_file_path(test_file_id, config.TIER_SSD_ENCODING)
    if ssd_path.exists():
        os.remove(ssd_path)
