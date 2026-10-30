"""
Exploratory Data Analysis script for Storage Access Traces and File Metadata.
Performs data quality checks, statistical profiling, and generates visual plots.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure reproducibility and set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "evaluation" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRACE_PATH = DATA_DIR / "storage_access_trace.csv"
META_PATH = DATA_DIR / "file_metadata.csv"

print("=" * 80)
print("PHASE 3: STORAGE ACCESS TRACE & METADATA EXPLORATION")
print("=" * 80)

# -------------------------------------------------------------
# 1. Dataset Loading & Schema Verification
# -------------------------------------------------------------
df_trace = pd.read_csv(TRACE_PATH)
df_meta = pd.read_csv(META_PATH)

print(f"\n[1] Trace Dataset Dimensions: {df_trace.shape[0]} rows x {df_trace.shape[1]} columns")
print(f"[1] Metadata Dataset Dimensions: {df_meta.shape[0]} rows x {df_meta.shape[1]} columns")

print("\n--- Trace Columns & Data Types ---")
for col, dtype in df_trace.dtypes.items():
    print(f"  {col:<25}: {dtype}")

print("\n--- Metadata Columns & Data Types ---")
for col, dtype in df_meta.dtypes.items():
    print(f"  {col:<25}: {dtype}")

print("\n--- Trace First 5 Rows ---")
print(df_trace.head())

print("\n--- Trace Last 5 Rows ---")
print(df_trace.tail())

print("\n--- Metadata First 5 Rows ---")
print(df_meta.head())

# Missing and duplicates
trace_missing = df_trace.isnull().sum()
meta_missing = df_meta.isnull().sum()
trace_dups = df_trace.duplicated().sum()
meta_dups = df_meta.duplicated().sum()

print(f"\nTrace missing values:\n{trace_missing}")
print(f"Trace duplicate rows: {trace_dups}")
print(f"Metadata missing values:\n{meta_missing}")
print(f"Metadata duplicate rows: {meta_dups}")

unique_files = df_trace['file_id'].nunique()
unique_apps = df_trace['application_id'].nunique() if 'application_id' in df_trace.columns else "N/A"
unique_ops = df_trace['operation'].unique().tolist()
unique_tiers = df_trace['tier'].unique().tolist()

print(f"\nUnique Files in Trace: {unique_files}")
print(f"Unique Applications: {unique_apps}")
print(f"Unique Operations: {unique_ops}")
print(f"Unique Storage Tiers: {unique_tiers}")

# -------------------------------------------------------------
# 2. Timestamp Analysis
# -------------------------------------------------------------
# Attempt parsing timestamp
# Check if timestamp is epoch or formatted string
print("\n[2] Timestamp Analysis:")
sample_ts = df_trace['timestamp'].iloc[0]
print(f"Sample timestamp: {sample_ts} (type: {type(sample_ts)})")

dt_series = pd.to_datetime(df_trace['timestamp'], errors='coerce')
min_ts = dt_series.min()
max_ts = dt_series.max()
ts_span_sec = (max_ts - min_ts).total_seconds()
print(f"Timestamp format: Datetime string")
print(f"Timestamp range : {min_ts} to {max_ts}")
print(f"Total time span : {ts_span_sec:.2f} seconds ({ts_span_sec/86400:.2f} days)")

# Monotonic check
is_monotonic = df_trace['timestamp'].is_monotonic_increasing
sorted_monotonic = df_trace['timestamp'].sort_values().is_monotonic_increasing
print(f"Timestamps naturally sorted (monotonic): {is_monotonic}")
print(f"Timestamps monotonic after explicit sort: {sorted_monotonic}")

# Events per day
days_series = (dt_series - dt_series.min()).dt.days
events_per_day = days_series.value_counts().sort_index()
print(f"\nEvents per day (0 to {days_series.max()}):\n{events_per_day}")

# -------------------------------------------------------------
# 3. Access Analysis
# -------------------------------------------------------------
total_events = len(df_trace)
read_count = (df_trace['operation'].str.upper() == 'READ').sum()
write_count = (df_trace['operation'].str.upper() == 'WRITE').sum()
other_op_count = total_events - (read_count + write_count)

read_pct = (read_count / total_events) * 100
write_pct = (write_count / total_events) * 100

ssd_count = (df_trace['tier'].str.upper() == 'SSD').sum()
hdd_count = (df_trace['tier'].str.upper() == 'HDD').sum()
ssd_pct = (ssd_count / total_events) * 100
hdd_pct = (hdd_count / total_events) * 100

avg_req_size = df_trace['request_size_bytes'].mean()
avg_resp_time = df_trace['response_time_us'].mean()
min_resp_time = df_trace['response_time_us'].min()
max_resp_time = df_trace['response_time_us'].max()

print("\n[3] Access Characteristics:")
print(f"  Total Events: {total_events}")
print(f"  Reads: {read_count} ({read_pct:.2f}%)")
print(f"  Writes: {write_count} ({write_pct:.2f}%)")
print(f"  SSD Accesses: {ssd_count} ({ssd_pct:.2f}%)")
print(f"  HDD Accesses: {hdd_count} ({hdd_pct:.2f}%)")
print(f"  Average Request Size: {avg_req_size:.2f} bytes ({avg_req_size/1024:.2f} KB)")
print(f"  Average Response Time: {avg_resp_time:.2f} us ({avg_resp_time/1000:.3f} ms)")
print(f"  Min Response Time: {min_resp_time} us | Max Response Time: {max_resp_time} us")

# -------------------------------------------------------------
# 4. File-Level Analysis
# -------------------------------------------------------------
file_grp = df_trace.groupby('file_id')
file_stats = pd.DataFrame({
    'total_accesses': file_grp.size(),
    'read_accesses': file_grp.apply(lambda g: (g['operation'].str.upper() == 'READ').sum()),
    'write_accesses': file_grp.apply(lambda g: (g['operation'].str.upper() == 'WRITE').sum()),
    'avg_response_time_us': file_grp['response_time_us'].mean(),
    'avg_request_size_bytes': file_grp['request_size_bytes'].mean(),
    'file_size_bytes': file_grp['file_size_bytes'].first()
})

print("\n[4] File-Level Analysis:")
print(f"Most frequently accessed files:\n{file_stats.sort_values(by='total_accesses', ascending=False)[['total_accesses', 'read_accesses', 'write_accesses']].head(5)}")
print(f"\nLeast frequently accessed files:\n{file_stats.sort_values(by='total_accesses', ascending=True)[['total_accesses', 'read_accesses', 'write_accesses']].head(5)}")
print(f"\nLargest files:\n{file_stats.sort_values(by='file_size_bytes', ascending=False)[['file_size_bytes', 'total_accesses']].head(5)}")
print(f"\nSmallest files:\n{file_stats.sort_values(by='file_size_bytes', ascending=True)[['file_size_bytes', 'total_accesses']].head(5)}")

# -------------------------------------------------------------
# 5. Ground Truth Analysis
# -------------------------------------------------------------
print("\n[5] Ground Truth Analysis (true_hotspot_next_6h):")
gt_counts = df_trace['true_hotspot_next_6h'].value_counts()
gt_pcts = df_trace['true_hotspot_next_6h'].value_counts(normalize=True) * 100
print(f"Event-level Hotspot Distribution:\n{gt_counts}")
print(f"Event-level Hotspot Percentages:\n{gt_pcts}")

# File-level hotspot distribution (whether file is marked hotspot in trace)
file_hotspot_any = df_trace.groupby('file_id')['true_hotspot_next_6h'].max()
print(f"\nUnique files flagged as hotspot at least once: {file_hotspot_any.sum()} / {len(file_hotspot_any)} ({file_hotspot_any.mean()*100:.1f}%)")

# -------------------------------------------------------------
# 6. Visualizations
# -------------------------------------------------------------
print("\n[6] Generating Visualizations...")

# 1. Accesses per file
plt.figure(figsize=(10, 5))
file_stats['total_accesses'].sort_values(ascending=False).reset_index(drop=True).plot(kind='line', color='#1f77b4', lw=2)
plt.title('File Access Frequency Distribution (Rank-Frequency)', fontsize=13, fontweight='bold')
plt.xlabel('File Index (Ranked by Accesses)', fontsize=11)
plt.ylabel('Total Access Count', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(FIG_DIR / "1_accesses_per_file.png", dpi=300)
plt.close()

# 2. Read vs Write distribution
plt.figure(figsize=(6, 5))
colors = ['#3498db', '#e74c3c']
plt.bar(['READ', 'WRITE'], [read_count, write_count], color=colors, width=0.5, edgecolor='black', alpha=0.85)
for i, count in enumerate([read_count, write_count]):
    pct = (count / total_events) * 100
    plt.text(i, count + (total_events * 0.01), f"{count:,}\n({pct:.1f}%)", ha='center', va='bottom', fontweight='bold')
plt.title('I/O Operation Distribution (Read vs Write)', fontsize=13, fontweight='bold')
plt.ylabel('Total Operations', fontsize=11)
plt.ylim(0, max(read_count, write_count) * 1.15)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(FIG_DIR / "2_read_vs_write_distribution.png", dpi=300)
plt.close()

# 3. HDD vs SSD distribution
plt.figure(figsize=(6, 5))
colors_tier = ['#95a5a6', '#2ecc71']
plt.bar(['HDD', 'SSD'], [hdd_count, ssd_count], color=colors_tier, width=0.5, edgecolor='black', alpha=0.85)
for i, count in enumerate([hdd_count, ssd_count]):
    pct = (count / total_events) * 100
    plt.text(i, count + (total_events * 0.01), f"{count:,}\n({pct:.1f}%)", ha='center', va='bottom', fontweight='bold')
plt.title('Storage Tier Access Distribution', fontsize=13, fontweight='bold')
plt.ylabel('Total Operations', fontsize=11)
plt.ylim(0, max(hdd_count, ssd_count) * 1.15)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(FIG_DIR / "3_hdd_vs_ssd_distribution.png", dpi=300)
plt.close()

# 4. File-size distribution
plt.figure(figsize=(9, 5))
sizes_mb = df_meta['file_size_bytes'] / (1024 * 1024) if 'file_size_bytes' in df_meta.columns else file_stats['file_size_bytes'] / (1024 * 1024)
sns.histplot(sizes_mb, bins=25, kde=True, color='#9b59b6', edgecolor='black')
plt.title('File Size Distribution (Metadata)', fontsize=13, fontweight='bold')
plt.xlabel('File Size (MB)', fontsize=11)
plt.ylabel('Number of Files', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(FIG_DIR / "4_file_size_distribution.png", dpi=300)
plt.close()

# 5. Response-time distribution
plt.figure(figsize=(9, 5))
# Use log scale or filter outlier for visualization if range is huge
resp_ms = df_trace['response_time_us'] / 1000.0
sns.histplot(resp_ms, bins=30, kde=True, color='#e67e22', edgecolor='black')
plt.title('Response Time Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Response Time (ms)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(FIG_DIR / "5_response_time_distribution.png", dpi=300)
plt.close()

# 6. Hotspot vs non-hotspot distribution
plt.figure(figsize=(6, 5))
hotspot_counts = df_trace['true_hotspot_next_6h'].value_counts().sort_index()
colors_hot = ['#34495e', '#e74c3c']
labels = ['Cold / Normal (0)', 'Hotspot (1)']
plt.bar(labels, hotspot_counts.values, color=colors_hot, width=0.5, edgecolor='black', alpha=0.85)
for i, count in enumerate(hotspot_counts.values):
    pct = (count / total_events) * 100
    plt.text(i, count + (total_events * 0.01), f"{count:,}\n({pct:.1f}%)", ha='center', va='bottom', fontweight='bold')
plt.title('Ground-Truth Hotspot Distribution (true_hotspot_next_6h)', fontsize=12, fontweight='bold')
plt.ylabel('Event Count', fontsize=11)
plt.ylim(0, max(hotspot_counts.values) * 1.15)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(FIG_DIR / "6_hotspot_vs_non_hotspot_distribution.png", dpi=300)
plt.close()

# 7. Access volume over time
plt.figure(figsize=(11, 5))
# Resample by 6-hour or 12-hour bins for clear trend
trace_copy = pd.DataFrame({'timestamp': dt_series})
trace_copy.set_index('timestamp', inplace=True)
resampled_counts = trace_copy.resample('6h').size()
resampled_counts.plot(kind='line', color='#2980b9', lw=2, marker='o', markersize=3)
plt.title('Storage Access Volume Over Time (6-Hour Bins)', fontsize=13, fontweight='bold')
plt.xlabel('Timeline', fontsize=11)
plt.ylabel('Total I/O Events', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(FIG_DIR / "7_access_volume_over_time.png", dpi=300)
plt.close()

print("[6] All 7 figures generated successfully in evaluation/figures/")

# -------------------------------------------------------------
# 7. Data Quality Checks
# -------------------------------------------------------------
print("\n[7] Data Quality Checks:")
neg_size = (df_trace['file_size_bytes'] < 0).sum()
neg_offset = (df_trace['offset_bytes'] < 0).sum()
neg_req = (df_trace['request_size_bytes'] <= 0).sum()
neg_resp = (df_trace['response_time_us'] < 0).sum()
invalid_tier_vals = (~df_trace['tier'].str.upper().isin(['HDD', 'SSD'])).sum()
invalid_ops_vals = (~df_trace['operation'].str.upper().isin(['READ', 'WRITE'])).sum()
malformed_ts = dt_series.isnull().sum()

print(f"  Negative file size: {neg_size}")
print(f"  Negative offset: {neg_offset}")
print(f"  Zero or negative request size: {neg_req}")
print(f"  Negative response time: {neg_resp}")
print(f"  Invalid tier values: {invalid_tier_vals}")
print(f"  Invalid operation values: {invalid_ops_vals}")
print(f"  Malformed timestamps: {malformed_ts}")
print(f"  Missing values in trace: {trace_missing.sum()}")
print(f"  Missing values in metadata: {meta_missing.sum()}")
print(f"  Duplicate events: {trace_dups}")

print("\nEDA script completed.")
