"""
Feature Visualization Script.
Generates exploratory plots for engineered features and the target variable.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

DATA_PATH = config.PROCESSED_DATASET_PATH
FIG_DIR = config.FIGURES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

print(f"[INFO] Loaded ML dataset with {len(df)} rows for feature visualization...")

# 1. recent_access_count distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["recent_access_count"], bins=40, kde=True, color="#2980b9", edgecolor="black")
plt.title("Distribution of recent_access_count (Observation Window)", fontsize=12, fontweight="bold")
plt.xlabel("Recent Access Count (t-6h to t)", fontsize=10)
plt.ylabel("Frequency (Window-File Instances)", fontsize=10)
plt.yscale("log")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_1_recent_access_count_dist.png", dpi=300)
plt.close()

# 2. future_access_count distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["future_access_count"], bins=40, kde=True, color="#e74c3c", edgecolor="black")
plt.title("Distribution of future_access_count (Target Variable)", fontsize=12, fontweight="bold")
plt.xlabel("Future Access Count (t to t+6h)", fontsize=10)
plt.ylabel("Frequency (Window-File Instances)", fontsize=10)
plt.yscale("log")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_2_future_access_count_dist.png", dpi=300)
plt.close()

# 3. recent_access_count vs future_access_count
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="recent_access_count",
    y="future_access_count",
    alpha=0.4,
    color="#8e44ad",
    edgecolor="none"
)
# Add trendline
sns.regplot(
    data=df,
    x="recent_access_count",
    y="future_access_count",
    scatter=False,
    color="#2c3e50",
    line_kws={"linewidth": 2, "linestyle": "--"}
)
plt.title("Recent Access Count vs. Future Access Count (Temporal Correlation)", fontsize=12, fontweight="bold")
plt.xlabel("Recent Access Count (t-6h to t)", fontsize=10)
plt.ylabel("Future Access Count (t to t+6h)", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_3_recent_vs_future_access_count.png", dpi=300)
plt.close()

# 4. recency_seconds vs future_access_count
plt.figure(figsize=(8, 6))
# Convert recency seconds to hours for intuitive axes
df["recency_hours"] = df["recency_seconds"] / 3600.0
sns.scatterplot(
    data=df,
    x="recency_hours",
    y="future_access_count",
    alpha=0.3,
    color="#d35400",
    edgecolor="none"
)
plt.title("Recency (Hours Since Last Access) vs. Future Access Count", fontsize=12, fontweight="bold")
plt.xlabel("Recency (Hours)", fontsize=10)
plt.ylabel("Future Access Count (t to t+6h)", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_4_recency_vs_future_access_count.png", dpi=300)
plt.close()

# 5. file_size_bytes vs future_access_count
plt.figure(figsize=(8, 6))
df["file_size_mb"] = df["file_size_bytes"] / (1024 * 1024)
sns.scatterplot(
    data=df,
    x="file_size_mb",
    y="future_access_count",
    alpha=0.3,
    color="#16a085",
    edgecolor="none"
)
plt.title("File Size (MB) vs. Future Access Count", fontsize=12, fontweight="bold")
plt.xlabel("File Size (MB)", fontsize=10)
plt.ylabel("Future Access Count (t to t+6h)", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_5_file_size_vs_future_access_count.png", dpi=300)
plt.close()

print(f"[SUCCESS] All 5 feature plots saved to {FIG_DIR}")
