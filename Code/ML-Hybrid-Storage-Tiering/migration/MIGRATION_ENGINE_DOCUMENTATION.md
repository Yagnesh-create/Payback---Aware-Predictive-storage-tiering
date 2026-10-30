# Migration Engine Documentation

## 1. Overview
The Migration Engine (`migration/migration_manager.py`) orchestrates the physical movement of files between the simulated storage tiers (`storage/HDD` and `storage/SSD`). It is strictly an *execution* engine—it makes no ML decisions itself, relying entirely on the `decision_engine.py` for recommendations.

## 2. Migration Workflow
To ensure data integrity, the engine uses a **Copy → Verify → Delete** workflow rather than a direct filesystem move.

1. **Validation**: Confirms file IDs, ensures the file is on the expected source tier, and ensures it doesn't already exist on the destination tier.
2. **Copy**: Duplicates the `.dat` file from source to destination.
3. **Verify**: Computes and compares SHA-256 checksums of both files.
4. **Delete**: If and only if the checksums match perfectly, the source file is deleted.

## 3. Checksum Verification
- The engine uses Python's `hashlib.sha256()`.
- Checksums are calculated by reading the file in chunks (4096 bytes) to remain memory-efficient.
- The `checksum_verified` boolean is recorded in the migration log.

## 4. Failure Handling
The engine is designed to fail safely:
- **Missing Source**: Migration is aborted (Status: `FAILED`).
- **Destination Conflict**: Migration is aborted (Status: `FAILED`).
- **Checksum Mismatch**: The newly copied destination file is immediately deleted to prevent corruption. The source file is strictly preserved (Status: `FAILED`).
- **Already on Destination**: If the file natively resides on the target tier, it is ignored (Status: `ALREADY_ON_DESTINATION`).
- Exceptions (e.g., `PermissionError`, disk full) are caught, the destination is cleaned up if a partial copy occurred, and the error string is returned.

## 5. Dry-Run Mode
Controlled via `DRY_RUN` in `config.py`.
- **True**: The system processes the entire ML and decision pipeline, generating logs and terminal outputs of what *would* happen, but bypasses the `migration_manager.py`. No files are moved.
- **False**: Active migration mode.

## 6. Migration Limits
Controlled via `MAX_MIGRATIONS_PER_RUN` in `config.py`.
- Provides a safety ceiling to prevent massive bulk migrations during testing, especially useful if payback thresholds are artificially lowered.

## 7. Logging
The `migration/migration_logger.py` records every attempted migration to `logs/migration_log.csv`.
Fields include:
- `migration_id`: Unique string combining a timestamp and short UUID.
- ML inputs: Heat score, predicted accesses, prediction decision.
- Execution metrics: Source/dest tiers, physical/logical sizes, success status, checksum validation, and execution duration.
