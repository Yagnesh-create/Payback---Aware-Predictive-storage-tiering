"""
Storage Manager (Phase 11) for ML-Driven Hybrid Storage Staging.

Manages the simulated local storage environment (storage/HDD and storage/SSD).
Ensures safe file paths, detects current file locations, and creates
simulated physical files up to a configured maximum physical size constraint.
"""

from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import os

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def get_safe_file_path(file_id: str, tier: int) -> Path:
    """
    Generate a safe absolute path for a simulated file.
    Prevents path traversal vulnerabilities.
    """
    # Sanitize file_id to prevent traversal
    safe_id = "".join(c for c in file_id if c.isalnum() or c in ('_', '-'))
    if not safe_id:
        raise ValueError("Invalid file ID")
    
    filename = f"{safe_id}.dat"
    if tier == config.TIER_HDD_ENCODING:
        return config.HDD_DIR / filename
    elif tier == config.TIER_SSD_ENCODING:
        return config.SSD_DIR / filename
    else:
        raise ValueError(f"Invalid tier encoding: {tier}")


def initialize_storage_directories():
    """Ensure that the physical storage simulation directories exist."""
    config.HDD_DIR.mkdir(parents=True, exist_ok=True)
    config.SSD_DIR.mkdir(parents=True, exist_ok=True)


def get_file_tier(file_id: str) -> Optional[int]:
    """
    Determine which tier a simulated file currently resides in.
    Returns TIER_HDD_ENCODING, TIER_SSD_ENCODING, or None if not found.
    If the file exists in BOTH tiers (invalid state), returns SSD by default
    and logs a warning, though this should be prevented by the migration engine.
    """
    safe_id = "".join(c for c in file_id if c.isalnum() or c in ('_', '-'))
    filename = f"{safe_id}.dat"
    
    on_ssd = (config.SSD_DIR / filename).exists()
    on_hdd = (config.HDD_DIR / filename).exists()
    
    if on_ssd and on_hdd:
        print(f"[WARNING] File {file_id} exists on BOTH tiers! Assuming SSD.")
        return config.TIER_SSD_ENCODING
    elif on_ssd:
        return config.TIER_SSD_ENCODING
    elif on_hdd:
        return config.TIER_HDD_ENCODING
    else:
        return None


def get_storage_state() -> Dict[str, float]:
    """Report the current state of the simulated storage directories."""
    state = {
        "hdd_files": 0,
        "ssd_files": 0,
        "hdd_physical_size_bytes": 0.0,
        "ssd_physical_size_bytes": 0.0,
        "total_files": 0,
    }
    
    if config.HDD_DIR.exists():
        for f in config.HDD_DIR.iterdir():
            if f.is_file():
                state["hdd_files"] += 1
                state["hdd_physical_size_bytes"] += f.stat().st_size
                
    if config.SSD_DIR.exists():
        for f in config.SSD_DIR.iterdir():
            if f.is_file():
                state["ssd_files"] += 1
                state["ssd_physical_size_bytes"] += f.stat().st_size
                
    state["total_files"] = state["hdd_files"] + state["ssd_files"]
    return state


def initialize_simulation_files(df_metadata: pd.DataFrame, initial_tiers: Dict[str, int]):
    """
    Create physical simulation files based on metadata.
    Uses SIMULATED_MAX_FILE_SIZE_MB to cap physical file sizes for practicality.
    
    Args:
        df_metadata: DataFrame containing 'file_id' and 'file_size_bytes'
        initial_tiers: Dictionary mapping file_id to initial tier (0=HDD, 1=SSD)
    """
    initialize_storage_directories()
    
    max_physical_bytes = int(config.SIMULATED_MAX_FILE_SIZE_MB * 1024 * 1024)
    
    created_count = 0
    for _, row in df_metadata.iterrows():
        file_id = row["file_id"]
        logical_size = int(row["file_size_bytes"])
        
        tier = initial_tiers.get(file_id, config.DEFAULT_TIER_ENCODING)
        filepath = get_safe_file_path(file_id, tier)
        
        # Determine physical size (capped)
        physical_size = min(logical_size, max_physical_bytes)
        
        # Only create if it doesn't already exist somewhere
        if get_file_tier(file_id) is None:
            # Create a sparse file or write null bytes
            try:
                with open(filepath, "wb") as f:
                    f.truncate(physical_size)
                created_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to create simulated file {filepath}: {e}")

    print(f"[INFO] Initialized {created_count} simulated physical files.")


def establish_initial_tiers() -> Dict[str, int]:
    """
    Determine the initial tier for each file based on the most recent 
    access event in the raw trace.
    """
    if not config.TRACE_DATASET_PATH.exists():
        return {}
        
    df_trace = pd.read_csv(config.TRACE_DATASET_PATH)
    # Sort chronologically
    df_trace = df_trace.sort_values(by="timestamp")
    
    # Take the last known tier for each file
    last_known_tiers = df_trace.groupby("file_id")["tier"].last()
    
    # Map tier string to encoding
    tier_map = {}
    for fid, tier_str in last_known_tiers.items():
        if str(tier_str).upper() == "SSD":
            tier_map[fid] = config.TIER_SSD_ENCODING
        else:
            tier_map[fid] = config.TIER_HDD_ENCODING
            
    return tier_map


if __name__ == "__main__":
    print("Testing Storage Manager...")
    initialize_storage_directories()
    
    df_meta = pd.read_csv(config.METADATA_DATASET_PATH)
    initial_tiers = establish_initial_tiers()
    
    print(f"Loaded metadata for {len(df_meta)} files.")
    print(f"Established initial tiers for {len(initial_tiers)} files.")
    
    print("\nInitializing simulated physical files...")
    initialize_simulation_files(df_meta, initial_tiers)
    
    state = get_storage_state()
    print("\nStorage State:")
    for k, v in state.items():
        if "size" in k:
            print(f"  {k}: {v / (1024*1024):.2f} MB")
        else:
            print(f"  {k}: {v}")
