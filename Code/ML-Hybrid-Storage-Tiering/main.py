"""
Main Pipeline Orchestrator for ML-Driven Hybrid Storage Staging.

Supports:
- Continuous execution: advances cycle-by-cycle as many times as you run it.
- Multi-cycle execution via --cycles N (e.g. python main.py --cycles 3).
- Direct cycle selection via --cycle C (e.g. python main.py --cycle 3).
- Real persistent physical storage tracking between cycles.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import pandas as pd

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent))
import config

from migration.storage_manager import (
    initialize_simulation_files,
    establish_initial_tiers,
    get_storage_state,
    get_file_tier,
)
from migration.decision_engine import (
    run_decision_pipeline,
    DECISION_MIGRATE,
)
from migration.migration_manager import (
    migrate_file,
    STATUS_SKIPPED,
)
from migration.migration_logger import log_migration

CYCLE_STATE_PATH = config.LOGS_DIR / "cycle_state.json"


def load_cycle_state() -> dict:
    """Load persistent cycle state from disk."""
    if CYCLE_STATE_PATH.exists():
        try:
            with open(CYCLE_STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"current_cycle": 1, "total_runs": 0}


def save_cycle_state(state: dict):
    """Save persistent cycle state to disk."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CYCLE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def reset_storage_environment():
    """Wipe simulated directories and re-seed from initial trace tiers."""
    print("\n" + "=" * 80)
    print("RESETTING STORAGE SIMULATION & CYCLE TRACKER")
    print("=" * 80)
    if config.STORAGE_DIR.exists():
        shutil.rmtree(config.STORAGE_DIR)
    config.HDD_DIR.mkdir(parents=True, exist_ok=True)
    config.SSD_DIR.mkdir(parents=True, exist_ok=True)
    
    df_meta = pd.read_csv(config.METADATA_DATASET_PATH)
    initial_tiers = establish_initial_tiers()
    initialize_simulation_files(df_meta, initial_tiers)
    
    save_cycle_state({"current_cycle": 1, "total_runs": 0})
    state = get_storage_state()
    print(f"[RESET COMPLETE] Storage seeded: HDD={state['hdd_files']} files, SSD={state['ssd_files']} files.")
    print("[RESET COMPLETE] Cycle counter reset to Cycle 1.\n")


def run_single_cycle(
    df_all: pd.DataFrame,
    df_meta: pd.DataFrame,
    cycle_num: int,
    total_available_cycles: int,
    target_timestamp: str,
    min_payback: float = 0.05,
    hotspot_threshold: int = config.HOTSPOT_THRESHOLD,
    dry_run: bool = config.DRY_RUN,
    max_migrations: int = config.MAX_MIGRATIONS_PER_RUN,
) -> dict:
    """
    Executes a single cycle in the migration pipeline.
    """
    print("\n" + "=" * 80)
    print(f"CYCLE {cycle_num} / {total_available_cycles} | TIMESTAMP: {target_timestamp}")
    print("=" * 80)

    # 1. Filter dataset for current timestamp
    df_current = df_all[df_all["prediction_timestamp"] == target_timestamp].copy()
    
    # 2. Ensure all simulation files exist physically on disk
    initial_tiers = {}
    for _, row in df_current.iterrows():
        initial_tiers[row["file_id"]] = int(row["current_tier"])
    initialize_simulation_files(df_meta, initial_tiers)

    # 3. Synchronize logical current_tier with REAL physical location on disk
    # This ensures that files moved in prior cycles persist in SSD!
    updated_from_disk = 0
    for idx, row in df_current.iterrows():
        phys_tier = get_file_tier(row["file_id"])
        if phys_tier is not None:
            if phys_tier != row["current_tier"]:
                updated_from_disk += 1
            df_current.at[idx, "current_tier"] = phys_tier

    state_before = get_storage_state()
    print(f"  Files evaluated        : {len(df_current)}")
    print(f"  Current Physical State : HDD={state_before['hdd_files']} files | SSD={state_before['ssd_files']} files")
    if updated_from_disk > 0:
        print(f"  (Preserved {updated_from_disk} file location(s) from previous migration cycles)")

    # 4. Decision Engine
    print(f"\n  [Decision Engine] Payback Threshold: {min_payback} | Hotspot Threshold: >= {hotspot_threshold}")
    df_decisions = run_decision_pipeline(
        df_current,
        threshold=hotspot_threshold,
        min_payback=min_payback,
    )

    candidates = df_decisions[df_decisions["decision"] == DECISION_MIGRATE].copy()
    predicted_hotspots = len(df_decisions[df_decisions["predicted_hotspot"] == 1])
    print(f"  Predicted Hotspots     : {predicted_hotspots}")
    print(f"  Approved for Migration : {len(candidates)}")

    # 5. Migration Execution
    success_count = 0
    failed_count = 0
    skipped_count = 0
    migrations_attempted = 0
    logical_sizes = dict(zip(df_meta["file_id"], df_meta["file_size_bytes"]))

    if dry_run:
        print("\n  [Execution Mode: DRY RUN] No files will be moved physically.")
        for _, row in candidates.iterrows():
            print(f"    [DRY RUN] Would migrate {row['file_id']} to SSD.")
    else:
        if len(candidates) > 0:
            print("\n  [Execution Mode: ACTIVE MIGRATION]")
            for _, row in candidates.iterrows():
                if migrations_attempted >= max_migrations:
                    print(f"    [LIMIT REACHED] Halting migrations at maximum ({max_migrations}).")
                    break
                    
                file_id = row["file_id"]
                source_tier = get_file_tier(file_id)
                if source_tier is None:
                    print(f"    [SKIPPED] {file_id}: File not found in physical simulation.")
                    skipped_count += 1
                    continue
                if source_tier == config.TIER_SSD_ENCODING:
                    # Already on SSD, skip
                    continue

                dest_tier = config.TIER_SSD_ENCODING
                print(f"    [MIGRATING] {file_id} (HDD -> SSD)...")
                result = migrate_file(file_id, source_tier, dest_tier)
                migrations_attempted += 1

                if result["status"] == "SUCCESS":
                    print(f"      SUCCESS: SHA-256 Checksum verified in {result['duration_seconds']:.4f}s")
                    success_count += 1
                else:
                    print(f"      FAILED: {result['error_message']}")
                    failed_count += 1

                log_size = int(logical_sizes.get(file_id, 0))
                decision_record = row.to_dict()
                log_migration(decision_record, result, log_size)
        else:
            print("\n  [Execution Mode: ACTIVE MIGRATION] No migrations required this cycle.")

    # 6. Verification
    state_after = get_storage_state()
    print("\n  " + "-" * 40)
    print("  CYCLE SUMMARY:")
    print(f"    Storage Before : HDD={state_before['hdd_files']} files, SSD={state_before['ssd_files']} files")
    print(f"    Storage After  : HDD={state_after['hdd_files']} files, SSD={state_after['ssd_files']} files")
    print(f"    Net Relocated  : {success_count} file(s) moved to SSD")
    print("  " + "-" * 40)

    return {
        "cycle": cycle_num,
        "timestamp": target_timestamp,
        "state_before": state_before,
        "state_after": state_after,
        "predicted_hotspots": predicted_hotspots,
        "approved": len(candidates),
        "success_count": success_count,
        "failed_count": failed_count,
    }


def run_full_pipeline(
    dataset_path: Path = config.TEST_DATASET_PATH,
    dry_run: bool = config.DRY_RUN,
    max_migrations: int = config.MAX_MIGRATIONS_PER_RUN,
    target_timestamp: str = None,
    min_payback: float = None,
    cycle: int = None,
    num_cycles: int = 1,
    hotspot_threshold: int = config.HOTSPOT_THRESHOLD,
):
    """
    Executes one or more migration cycles across chronological observation windows.
    """
    df_all = pd.read_csv(dataset_path)
    df_meta = pd.read_csv(config.METADATA_DATASET_PATH)
    timestamps = sorted(df_all["prediction_timestamp"].unique())
    total_available_cycles = len(timestamps)

    effective_payback = min_payback if min_payback is not None else 0.05

    # If a specific timestamp is passed directly (e.g. from tests)
    if target_timestamp is not None:
        idx = timestamps.index(target_timestamp) if target_timestamp in timestamps else 0
        run_single_cycle(
            df_all=df_all,
            df_meta=df_meta,
            cycle_num=idx + 1,
            total_available_cycles=total_available_cycles,
            target_timestamp=target_timestamp,
            min_payback=effective_payback,
            hotspot_threshold=hotspot_threshold,
            dry_run=dry_run,
            max_migrations=max_migrations,
        )
        return

    # Cycle-based orchestration
    state_tracker = load_cycle_state()
    start_cycle = cycle if cycle is not None else state_tracker.get("current_cycle", 1)

    initial_storage = get_storage_state()
    total_moved = 0
    cycle_results = []

    current_c = start_cycle
    for i in range(num_cycles):
        # Normalize cycle to available timestamps (1-indexed modulo)
        normalized_c = ((current_c - 1) % total_available_cycles) + 1
        ts = timestamps[normalized_c - 1]

        if current_c > total_available_cycles:
            print(f"\n[CONTINUOUS MODE] Cycling past test window #{total_available_cycles}. Advancing through cycle {normalized_c}.")

        res = run_single_cycle(
            df_all=df_all,
            df_meta=df_meta,
            cycle_num=normalized_c,
            total_available_cycles=total_available_cycles,
            target_timestamp=ts,
            min_payback=effective_payback,
            hotspot_threshold=hotspot_threshold,
            dry_run=dry_run,
            max_migrations=max_migrations,
        )
        cycle_results.append(res)
        total_moved += res["success_count"]
        current_c += 1

    # Update persistent state
    next_cycle = ((current_c - 1) % total_available_cycles) + 1
    state_tracker["current_cycle"] = next_cycle
    state_tracker["total_runs"] = state_tracker.get("total_runs", 0) + num_cycles
    save_cycle_state(state_tracker)

    # Multi-cycle summary
    if num_cycles > 1:
        final_storage = get_storage_state()
        print("\n" + "=" * 80)
        print(f"MULTI-CYCLE EXECUTION COMPLETE ({num_cycles} cycles executed)")
        print("=" * 80)
        print(f"  Storage at Start : HDD={initial_storage['hdd_files']} files | SSD={initial_storage['ssd_files']} files")
        print(f"  Storage at End   : HDD={final_storage['hdd_files']} files | SSD={final_storage['ssd_files']} files")
        print(f"  Total Migrations : {total_moved} file(s) moved across all cycles")
        print(f"  Next Cycle       : Cycle {next_cycle} ({timestamps[next_cycle - 1]})")
        print("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ML-Driven Hybrid Storage Staging and Hotspot Migration Orchestrator"
    )
    parser.add_argument(
        "-c", "--cycle",
        type=int,
        default=None,
        help="Run a specific chronological cycle (1 to 8). If omitted, advances automatically from previous run.",
    )
    parser.add_argument(
        "-n", "--cycles",
        type=int,
        default=1,
        help="Number of consecutive cycles to execute in sequence (e.g. -n 3).",
    )
    parser.add_argument(
        "--payback",
        type=float,
        default=0.05,
        help="Minimum payback ratio threshold (default: 0.05 for active evaluation demonstration).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=config.HOTSPOT_THRESHOLD,
        help=f"Hotspot threshold access count (default: {config.HOTSPOT_THRESHOLD}).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset storage directories and cycle counter to clean initial state.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform predictions and decisions without executing physical file moves.",
    )
    parser.add_argument(
        "--list-cycles",
        action="store_true",
        help="List all available chronological cycles and their timestamps.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.reset:
        reset_storage_environment()
        sys.exit(0)

    if args.list_cycles:
        df_all = pd.read_csv(config.TEST_DATASET_PATH)
        ts_list = sorted(df_all["prediction_timestamp"].unique())
        print("\nAvailable Chronological Cycles:")
        for idx, ts in enumerate(ts_list):
            print(f"  Cycle {idx + 1}: {ts}")
        print()
        sys.exit(0)

    run_full_pipeline(
        dry_run=args.dry_run,
        min_payback=args.payback,
        cycle=args.cycle,
        num_cycles=args.cycles,
        hotspot_threshold=args.threshold,
    )
