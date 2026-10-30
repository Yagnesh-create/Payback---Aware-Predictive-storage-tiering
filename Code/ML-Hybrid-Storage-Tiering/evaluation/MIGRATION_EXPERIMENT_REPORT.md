# Migration Experiment Report (Phases 11-15)

## 1. Experiment Setup
- **Dataset**: `data/test.csv` (1,600 temporal rows across 200 files)
- **Target Timestamp**: `2026-01-14 12:00:00`
- **Execution Mode**: Active Migration (`DRY_RUN = False`)
- **Machine Learning Model**: LinearRegression (previously trained in Phase 6)
- **Hotspot Threshold**: >= 8 predicted accesses in next 6h
- **Payback Threshold Constraint Modification**: The default payback threshold is 1.0 (benefit must exceed cost). At the target timestamp, the single HDD hotspot had a payback ratio of ~0.30. To test the active physical migration engine, the threshold was temporarily overridden during the E2E test to force the migration of this file.

## 2. Evaluation Results
The system successfully evaluated 200 files at the target timestamp.

| Metric | Count |
| :--- | :--- |
| **Files Evaluated** | 200 |
| **Predicted Hotspots** | 27 |
| **Migration Recommendations** | 1 |
| **Actual Migrations Attempted** | 1 |
| **Successful Migrations** | 1 |
| **Failed/Skipped Migrations** | 0 |

## 3. Storage State Distribution

**BEFORE MIGRATION:**
- HDD Files: 164
- SSD Files: 36

**AFTER MIGRATION:**
- HDD Files: 163
- SSD Files: 37

## 4. Execution Metrics
- **Migration Engine Status**: The selected candidate (`file_0074`) was successfully copied from `storage/HDD` to `storage/SSD`.
- **Integrity**: Checksum matching verified successfully. Source file on HDD was deleted post-verification.
- **Migration Duration**: The physical simulation file was successfully transferred and verified in ~0.029 seconds. Note that this is the local simulated execution duration, NOT the simulated latency calculation used by the payback engine (which estimates savings in microseconds).

## 5. End-to-End Test Suite Outcomes
A comprehensive test suite (`tests/test_end_to_end.py`) encompassing 27 individual assertions validated the complete integration of Phase 11-15 modules.

**Highlights:**
- Directory initialization and file detection operate safely within the configured simulated workspace boundaries.
- Both forward (HDD → SSD) and reverse (SSD → HDD) migrations function correctly with strict integrity verification.
- Failed checksums correctly trigger a safe abort that deletes the corrupted destination file while preserving the source.
- Payback calculation mathematically integrates successfully, scaling by logical file sizes independently of physical disk caps.
- Migration events correctly populate `logs/migration_log.csv` with 19 schema-compliant telemetry data points per migration.

## 6. Limitations
1. **Simulation File Capping**: To prevent disk space exhaustion, the system caps the physical size of the `.dat` files at 1.0 MB by default (`SIMULATED_MAX_FILE_SIZE_MB`). The migration duration is artificially fast as a result.
2. **Reverse Migration Unused**: While the `migration_manager.py` supports reverse migrations (SSD → HDD), the `decision_engine.py` currently only focuses on upward tiering (HDD → SSD). Eviction policies are not yet implemented.
3. **Threshold Sensitivity**: True test-set hotspots rarely meet the default `MIN_PAYBACK_RATIO` of 1.0 due to high physical migration bandwidth costs.
