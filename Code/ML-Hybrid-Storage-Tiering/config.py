"""
Configuration Module for ML-Driven Hybrid Storage Staging and Hotspot Migration.

Contains configurable constants, paths, and baseline system parameters.
All simulation parameters are clearly identified as initial configuration
choices, NOT experimentally proven optimal values.
"""

from pathlib import Path

# ==========================================
# 1. Base Paths & Directory Structure
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
HDD_DIR = STORAGE_DIR / "HDD"
SSD_DIR = STORAGE_DIR / "SSD"

MODELS_DIR = BASE_DIR / "models"
MONITOR_DIR = BASE_DIR / "monitor"
FEATURES_DIR = BASE_DIR / "features"
ML_DIR = BASE_DIR / "ml"
MIGRATION_DIR = BASE_DIR / "migration"
FEEDBACK_DIR = BASE_DIR / "feedback"
EVALUATION_DIR = BASE_DIR / "evaluation"
FIGURES_DIR = EVALUATION_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"
DASHBOARD_DIR = BASE_DIR / "dashboard"

# Dataset file paths
TRACE_DATASET_PATH = DATA_DIR / "storage_access_trace.csv"
METADATA_DATASET_PATH = DATA_DIR / "file_metadata.csv"
PROCESSED_DATASET_PATH = DATA_DIR / "ml_training_dataset.csv"

# Phase 5 Split & Artifact Paths
TRAIN_DATASET_PATH = DATA_DIR / "train.csv"
VAL_DATASET_PATH = DATA_DIR / "validation.csv"
TEST_DATASET_PATH = DATA_DIR / "test.csv"
SPLIT_METADATA_PATH = DATA_DIR / "data_split_metadata.json"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
MODEL_PATH = MODELS_DIR / "hotspot_predictor.pkl"

# ==========================================
# 2. Machine Learning Configuration
# ==========================================
RANDOM_SEED = 42

# Temporal Window Parameters (in hours)
WARMUP_HOURS = 12
RECENT_WINDOW_HOURS = 6
PREVIOUS_WINDOW_HOURS = 6
FUTURE_WINDOW_HOURS = 6
STEP_WINDOW_HOURS = 6

# Temporal Split Ratios (Chronological)
TRAIN_SPLIT_RATIO = 0.70
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15

# Feature names (strictly available BEFORE the prediction point)
FEATURE_COLUMNS = [
    "recent_access_count",
    "previous_access_count",
    "recency_seconds",
    "burstiness",
    "file_size_bytes",
    "read_ratio",
    "write_ratio",
    "average_request_size",
    "current_tier",
]

# Target column (future access count within the future window)
TARGET_COLUMN = "future_access_count"

# Evaluation-only ground truth column (MUST NEVER be used as an input feature)
EVAL_GROUND_TRUTH_COLUMN = "eval_true_hotspot_next_6h"

# Hotspot classification threshold (>= this value in future window = hotspot)
HOTSPOT_THRESHOLD = 8

# Tier numerical encodings
TIER_HDD_ENCODING = 0
TIER_SSD_ENCODING = 1

# Recency default for files never accessed in observation history (seconds = 12h)
MAX_RECENCY_SECONDS = 43200.0

# ==========================================
# 3. Heat-Map Configuration (Phase 7)
# ==========================================
# Heat score formula: H = alpha * frequency_score + beta * recency_score
# NOTE: These are initial configuration choices, NOT experimentally
#       proven optimal weights. They can be tuned without modifying
#       the algorithm implementation.
HEAT_FREQUENCY_WEIGHT = 0.7   # alpha
HEAT_RECENCY_WEIGHT = 0.3     # beta

# ==========================================
# 4. Storage Tier Simulation Parameters
# ==========================================
# NOTE: All latency and bandwidth values below are SIMULATION PARAMETERS.
# They are NOT measurements from physical hardware. They represent
# plausible order-of-magnitude estimates for algorithm evaluation.

# Simulated baseline latency per I/O operation (in microseconds)
HDD_READ_LATENCY_US = 10000.0    # 10 ms = 10000 us
SSD_READ_LATENCY_US = 200.0      # 0.2 ms = 200 us

# Legacy ms-based latency constants (kept for backward compatibility)
HDD_LATENCY_MS = HDD_READ_LATENCY_US / 1000.0   # 10.0 ms
SSD_LATENCY_MS = SSD_READ_LATENCY_US / 1000.0    # 0.2 ms

# Simulated SSD capacity constraint in Megabytes (MB)
SSD_CAPACITY_MB = 2048.0

# Simulated migration bandwidth (bytes per second)
# 100 MB/s is a reasonable estimate for a SATA SSD sequential write
MIGRATION_BANDWIDTH_BYTES_PER_SEC = 100 * 1024 * 1024  # 100 MB/s

# ==========================================
# 5. Migration Decision Parameters (Phase 9–10)
# ==========================================
# Minimum payback ratio required to approve a migration.
# payback_ratio = expected_benefit / migration_cost
# A value of 1.0 means benefit must at least equal the cost.
# NOTE: This is an initial policy threshold, NOT a scientifically
#       optimal value.
# MIN_PAYBACK_RATIO = 1.0
MIN_PAYBACK_RATIO = 1

# ==========================================
# 6. Storage Simulation Parameters (Phase 11-15)
# ==========================================
# Maximum physical size of a simulated file on disk.
# If logical metadata size exceeds this, the physical file is capped.
SIMULATED_MAX_FILE_SIZE_MB = 1.0  # Keep it very small (1MB) to prevent disk fill-up during tests

# Default tier if none is specified or found
DEFAULT_TIER_ENCODING = TIER_HDD_ENCODING

# ==========================================
# 7. Execution and Logging (Phase 13-14)
# ==========================================
# When True, the pipeline generates recommendations but DOES NOT move files.
DRY_RUN = False

# Safety limit for maximum physical migrations per run
MAX_MIGRATIONS_PER_RUN = 50

# Log file paths
MIGRATION_LOG_PATH = LOGS_DIR / "migration_log.csv"
ACCESS_LOG_PATH = LOGS_DIR / "access_log.csv"
FEEDBACK_LOG_PATH = LOGS_DIR / "feedback_log.csv"

# ==========================================
# 8. Feedback Configuration (Phase 16)
# ==========================================
# NOTE: These are experimental configuration parameters, NOT
#       scientifically proven optimal values. They control the
#       adaptive heat-score feedback mechanism that adjusts
#       migration decisions based on observed prediction errors.

# How strongly the feedback signal adjusts the heat score.
# Range: (0, 1]. Higher values make the system more responsive
# to prediction errors but also more volatile.
FEEDBACK_WEIGHT = 0.1

# Normalisation scale for the tanh bounding function.
# feedback_signal / FEEDBACK_SCALE is passed to tanh().
# Larger values make the bounding more lenient (slower saturation).
# Smaller values make it saturate faster, limiting extreme adjustments.
FEEDBACK_SCALE = 5.0
