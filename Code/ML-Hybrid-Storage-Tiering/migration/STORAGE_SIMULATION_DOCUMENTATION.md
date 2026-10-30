# Storage Simulation Documentation

## 1. Why Simulate Storage?
This project is an ML-driven hybrid storage tiering system. To provide a reproducible, hardware-independent development and testing environment, we simulate physical HDD and SSD storage using local directories. This avoids:
- Modifying actual system hardware partitioning.
- The need for specialized SSD/HDD configurations on student laptops.
- Accidental data loss outside the project scope.
- Platform-specific filesystem calls.

## 2. Directory Structure
The simulated storage environment resides entirely within:
```text
storage/
├── HDD/    # Represents the slow, high-capacity tier
└── SSD/    # Represents the fast, low-latency tier
```

## 3. File Representation
The raw `storage_access_trace.csv` references files by `file_id` (e.g., `file_0001`). 
To simulate physical storage:
- The system generates a sanitized `.dat` file (e.g., `file_0001.dat`) in one of the tier directories.
- This mapping ensures that abstract IDs are translated into safe, bounded file paths, strictly confined to the `storage/` directory.

## 4. Initial Tier Assignment
To maintain physical consistency with the logical dataset:
- The `storage_manager.py` analyzes the raw trace (`storage_access_trace.csv`).
- It determines the *most recent* recorded tier for each file.
- If a file exists in the metadata but has no trace events, it defaults to HDD (`TIER_HDD_ENCODING`).

## 5. Physical Simulation-Size Limitations
The `file_metadata.csv` specifies logical file sizes (some exceeding hundreds of MBs). To ensure the simulation remains practical on standard laptops and does not rapidly fill disk space:
- **`SIMULATED_MAX_FILE_SIZE_MB`**: Configured in `config.py` (default: 1.0 MB).
- When a file is initialized in `storage/`, its physical size is capped at this maximum.
- **Logical Independence**: The *logical* size (from metadata) is still used by the decision engine for calculating migration costs. The physical cap applies *only* to the `.dat` file footprint on disk.

## 6. Safety and Limitations
- **Path Traversal Protection**: All `file_id`s are sanitized, stripping slashes and non-alphanumeric characters before path construction.
- **Not Real Performance**: The simulation handles *file presence and location* but does not measure real HDD/SSD IOPS or latency. Latency gains are mathematically estimated in the payback calculator based on configured assumptions.
