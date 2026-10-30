# Dataset Analysis Report: Storage Access Trace & File Metadata

> **Important Dataset Clarification**:  
> The primary dataset analyzed herein is a **controlled synthetic trace dataset** designed specifically for researching and evaluating hybrid storage tiering algorithms under reproducible conditions (generated with random seed 42). It is not captured from a live proprietary enterprise production SAN/NAS environment.

---

## 1. Executive Summary & Dimensions
The dataset consists of two primary relational CSV files located in the `data/` directory:

| Dataset File | Row Count | Column Count | Memory Footprint (approx.) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`storage_access_trace.csv`** | **28,000** | **11** | ~2.3 MB | Event-by-event storage I/O trace over 14 simulated days |
| **`file_metadata.csv`** | **200** | **4** | ~6.6 KB | Static file metadata (size, owning application, file type) |

---

## 2. Schema Specification & Data Types

### 2.1 Storage Access Trace (`storage_access_trace.csv`)
| Column Name | Inferred Type | Python/Pandas Type | Description & Domain |
| :--- | :--- | :--- | :--- |
| `event_id` | Integer | `int64` | Unique sequential identifier for each I/O event |
| `timestamp` | Datetime (string) | `str` / `datetime64[ns]` | ISO timestamp format (`YYYY-MM-DD HH:MM:SS`) |
| `file_id` | Identifier (string) | `str` | Target file identifier (`file_0001` to `file_0200`) |
| `operation` | Categorical | `str` | I/O operation type (`READ`, `WRITE`) |
| `file_size_bytes` | Integer | `int64` | Total size of the file at time of access |
| `offset_bytes` | Integer | `int64` | Starting byte offset within the file |
| `request_size_bytes`| Integer | `int64` | Size of the requested I/O block in bytes |
| `response_time_us` | Integer | `int64` | Latency/response time in microseconds ($\mu\text{s}$) |
| `tier` | Categorical | `str` | Servicing physical storage tier (`HDD`, `SSD`) |
| `application_id` | Categorical | `str` | Workload source application (`app_01` to `app_10`) |
| `true_hotspot_next_6h` | Binary Flag | `int64` | **Ground truth** future label ($1 = \text{hotspot}, 0 = \text{normal}$) |

> **Strict Data Leakage Guard**:  
> `true_hotspot_next_6h` is ground truth representing future 6-hour access behavior. As mandated in the project specification, this column **MUST NOT** be used as an input feature for ML training.

### 2.2 File Metadata (`file_metadata.csv`)
| Column Name | Inferred Type | Python/Pandas Type | Description |
| :--- | :--- | :--- | :--- |
| `file_id` | Identifier (string) | `str` | Primary key linking to trace records |
| `file_size_bytes` | Integer | `int64` | Static file size in bytes |
| `application_id` | Categorical | `str` | Primary owner application identifier |
| `file_type` | Categorical | `str` | Semantic workload classification (`log`, `db`, `document`, etc.) |

---

## 3. Data Quality & Integrity Audit
An exhaustive audit across all 28,000 trace rows and 200 metadata entries confirmed high integrity:

| Data Quality Check | Target Requirement | Audit Result | Status |
| :--- | :--- | :--- | :--- |
| **Missing / Null Values** | 0 nulls across all fields | 0 nulls in trace; 0 in metadata | **PASSED** |
| **Duplicate Rows** | No identical duplicate rows | 0 duplicates found | **PASSED** |
| **Timestamp Monotonicity** | Ascending chronological order | Monotonic (`is_monotonic = True`) | **PASSED** |
| **Negative Sizes / Offsets** | $\ge 0$ | 0 negative values | **PASSED** |
| **Zero/Negative Requests** | $> 0$ | 0 non-positive request sizes | **PASSED** |
| **Negative Latencies** | $\ge 0$ | 0 negative response times | **PASSED** |
| **Invalid Operation Categories**| Only `READ` or `WRITE` | Exactly 2 unique values (`READ`, `WRITE`) | **PASSED** |
| **Invalid Storage Tiers** | Only `HDD` or `SSD` | Exactly 2 unique values (`HDD`, `SSD`) | **PASSED** |

---

## 4. Temporal & Access Pattern Analysis

### 4.1 Timestamp Dynamics
- **Start Timestamp**: `2026-01-01 00:00:01`
- **End Timestamp**: `2026-01-14 23:59:40`
- **Total Duration**: `14.00 days` (1,209,579 seconds)
- **Trace Consistency**: Exactly **2,000 events per day** consistently across all 14 days (Days 0 through 13).

### 4.2 Workload Access Breakdown
- **Total I/O Events**: **28,000**
- **Read Operations**: **23,003** (**82.15%**)
- **Write Operations**: **4,997** (**17.85%**)
  - *Observation*: The workload is read-heavy (~4.6:1 read-to-write ratio), characteristic of typical read-dominant staging workloads.
- **Storage Tier Distribution**:
  - **SSD Accesses**: **15,420** (**55.07%**)
  - **HDD Accesses**: **12,580** (**44.93%**)
- **I/O Request Sizes**:
  - Average: **25,355.26 bytes** (~24.76 KB)
  - Min: 4,096 bytes (4 KB block aligned)
  - Max: 65,536 bytes (64 KB)
- **Response Times**:
  - Mean: **525.60 $\mu\text{s}$** (0.526 ms)
  - Minimum: **20 $\mu\text{s}$**
  - Maximum: **1,917 $\mu\text{s}$** (1.917 ms)

---

## 5. File-Level & Entity Profile

### 5.1 Entities
- **Unique Files**: **200** files (`file_0001` through `file_0200`).
- **Unique Applications**: **10** distinct applications (`app_01` through `app_10`).

### 5.2 Access Skewness (Hotspot Behavior)
Access distribution exhibits a Pareto-like skew:
- **Top 5 Most Accessed Files**:
  1. `file_0066`: 691 accesses (569 reads, 122 writes)
  2. `file_0083`: 675 accesses (570 reads, 105 writes)
  3. `file_0120`: 670 accesses (543 reads, 127 writes)
  4. `file_0033`: 660 accesses (536 reads, 124 writes)
  5. `file_0014`: 616 accesses (512 reads, 104 writes)
- **Least Accessed Files**:
  - `file_0011`: 2 accesses
  - `file_0151`: 2 accesses
  - `file_0199`: 4 accesses
  - `file_0121`: 4 accesses
  - `file_0102`: 8 accesses

### 5.3 File Capacities
- **Largest Files**:
  - `file_0158`: 208,898,775 bytes (~199.2 MB)
  - `file_0143`: 208,177,098 bytes (~198.5 MB)
  - `file_0139`: 206,095,249 bytes (~196.5 MB)
- **Smallest Files**:
  - `file_0104`: 1,548,045 bytes (~1.48 MB)
  - `file_0167`: 4,767,069 bytes (~4.55 MB)

---

## 6. Ground-Truth Analysis (`true_hotspot_next_6h`)
- **Event-Level Distribution**:
  - Class `0` (Normal / Cold): **15,002** records (**53.58%**)
  - Class `1` (Future Hotspot): **12,998** records (**46.42%**)
- **File-Level Distribution**:
  - Out of 200 total files, **143 files (71.5%)** experience at least one hotspot window during the 14-day timeline.
- **Balance**: The dataset has an exceptionally balanced binary target distribution (~54% vs ~46%), preventing severe class imbalance distortions when evaluating hotspot predictions.

---

## 7. Generated Exploratory Visualizations
High-resolution exploratory charts have been generated and archived under [`evaluation/figures/`](file:///d:/THEJA%20VIT/OS%20operating%20system/Lab%20Project/Code/ML-Hybrid-Storage-Tiering/evaluation/figures):

1. **`1_accesses_per_file.png`**: Rank-frequency curve showing pronounced skew towards hotspot files.
2. **`2_read_vs_write_distribution.png`**: Breakdown of read operations (82.15%) vs. write operations (17.85%).
3. **`3_hdd_vs_ssd_distribution.png`**: Breakdown of current physical tier assignments (55.07% SSD vs. 44.93% HDD).
4. **`4_file_size_distribution.png`**: Histogram depicting file size distribution across the 200 files.
5. **`5_response_time_distribution.png`**: Latency profile highlighting fast SSD servicing vs. higher-latency HDD responses.
6. **`6_hotspot_vs_non_hotspot_distribution.png`**: Ground-truth class balance comparison.
7. **`7_access_volume_over_time.png`**: Aggregated I/O timeline in 6-hour intervals showing steady periodic activity over the 14 days.

---

## 8. Summary of Findings for Phase 4 Preparation
1. The synthetic trace exhibits realistic tiering attributes: read dominance, access skewness, and clear latency differentials between SSD and HDD.
2. Data quality is pristine: zero missing values, zero corrupted records, and monotonic timestamps.
3. Feature engineering in Phase 4 can cleanly calculate rolling window stats (e.g. 6-hour or 1-day windows) without requiring data cleansing.
4. The strict separation of `true_hotspot_next_6h` will be rigorously maintained.
