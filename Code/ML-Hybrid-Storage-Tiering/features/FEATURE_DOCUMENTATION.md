# Feature Documentation: ML-Driven Hybrid Storage Staging & Hotspot Migration

This document specifies the schema, mathematical formulation, observation windows, and predictive rationale for all engineered input features, the regression target, and evaluation-only ground-truth labels.

---

## 1. Temporal Prediction Framework & Leakage Safeguard

```
             PAST OBSERVATION WINDOW                PREDICTION POINT (t)        FUTURE PREDICTION WINDOW
   [----------------------------------------------]          │          [--------------------------------]
   t - 12h                                  t - 6h           │          t                               t + 6h
      │                                       │              │                          │
      └────── Previous Observation Window ────┘              │                          │
                      (t - 12h to t - 6h)                    │                          │
                                                             │                          │
                                              └─ Recent ─────┘                          │
                                                 (t - 6h to t)                          │
                                                             │                          │
                                               FEATURES EXTRACTED                       │
                                             (Strictly for dt < t)                      │
                                                                                        │
                                                                           TARGET & EVALUATION GROUND TRUTH
                                                                                 (Counted for t <= dt < t + 6h)
```

- **Strict Temporal Boundary**: No event occurring at or after prediction timestamp $t$ is ever included in the computation of any feature.
- **Warmup Period**: Feature extraction begins after 12 hours of historical warmup (`2026-01-01 12:00:00`) to guarantee both previous and recent observation windows are fully populated.
- **Sliding Step**: New prediction points are sampled every 6 hours across the 14-day duration, producing 54 distinct prediction timestamps.

---

## 2. Feature Definitions & Schema

### Feature 1: `recent_access_count`
- **Meaning**: Total number of I/O operations (reads + writes) directed to the file during the immediate past window.
- **Observation Window**: $[t - 6\text{h}, t)$ (Past 6 hours).
- **Calculation**: $\sum_{e \in \mathcal{E}} \mathbb{I}(\text{file}(e) = f \land t - 6\text{h} \le \text{timestamp}(e) < t)$.
- **Directionality**: Higher values indicate higher recent activity.
- **Predictive Value**: Reflects temporal locality; in storage systems, files accessed heavily in the recent past exhibit strong inertia to remain active in the near future ($r = +0.6504$).
- **Availability Before $t$**: **Yes** — derived strictly from historical trace records prior to $t$.

---

### Feature 2: `previous_access_count`
- **Meaning**: Total number of I/O operations directed to the file in the observation window preceding the recent window.
- **Observation Window**: $[t - 12\text{h}, t - 6\text{h})$ (12 to 6 hours before $t$).
- **Calculation**: $\sum_{e \in \mathcal{E}} \mathbb{I}(\text{file}(e) = f \land t - 12\text{h} \le \text{timestamp}(e) < t - 6\text{h})$.
- **Directionality**: Higher values indicate persistent long-term activity.
- **Predictive Value**: Differentiates between short transient bursts (high recent, low previous) and sustained workloads (high recent, high previous) ($r = +0.3117$).
- **Availability Before $t$**: **Yes** — strictly historical.

---

### Feature 3: `recency_seconds`
- **Meaning**: Elapsed time (in seconds) between the file's most recent I/O access and the prediction timestamp $t$.
- **Observation Window**: $[0, t)$ (All history up to $t$).
- **Calculation**: $\text{recency\_seconds} = (t - \max \{\text{timestamp}(e) : \text{file}(e) = f \land \text{timestamp}(e) < t\}).\text{total\_seconds}()$.
  *Default*: If never accessed prior to $t$, assigned maximum value (`config.MAX_RECENCY_SECONDS = 43,200.0` s).
- **Directionality**: **Lower values** indicate more recent activity.
- **Predictive Value**: Fundamental principle of Least Recently Used (LRU) caching and storage tiering; files touched seconds ago are far more likely to be accessed again than files idle for hours ($r = -0.1671$).
- **Availability Before $t$**: **Yes**.

---

### Feature 4: `burstiness`
- **Meaning**: Numerical metric quantifying how temporally concentrated or clustered access arrivals are.
- **Observation Window**: $[t - 6\text{h}, t)$ (Recent window).
- **Calculation**: Goh-Barabasi normalized burstiness parameter:
  $$B = \frac{\sigma_{\Delta t} - \mu_{\Delta t}}{\sigma_{\Delta t} + \mu_{\Delta t}}$$
  where $\Delta t$ is the vector of inter-arrival time intervals (in seconds) between successive accesses to file $f$.
  - $B \in [-1.0, 1.0]$.
  - $B = -1$: Perfectly periodic/regular access pacing ($\sigma_{\Delta t} = 0$).
  - $B = 0$: Memoryless exponential/Poisson arrivals ($\sigma_{\Delta t} = \mu_{\Delta t}$).
  - $B > 0$: Bursty, clustered arrival spikes ($\sigma_{\Delta t} > \mu_{\Delta t}$).
  - Defaults to $0.0$ if access count $N \le 1$.
- **Directionality**: Higher values indicate higher temporal clustering.
- **Predictive Value**: Identifies batch-oriented workloads and micro-bursts that may quickly subside versus continuous smooth traffic.
- **Availability Before $t$**: **Yes**.

---

### Feature 5: `file_size_bytes`
- **Meaning**: Total capacity of the file in bytes.
- **Observation Window**: Static metadata / current file capacity.
- **Calculation**: Extracted from `file_metadata.csv`.
- **Directionality**: Neutral.
- **Predictive Value**: Smaller files (e.g. metadata tables, index logs) often exhibit higher relative access frequencies, while large files impose greater migration penalties. Essential for cost-benefit computation.
- **Availability Before $t$**: **Yes**.

---

### Feature 6: `read_ratio`
- **Meaning**: Proportion of operations on the file that were read requests.
- **Observation Window**: $[t - 6\text{h}, t)$ (with fallback to past 12h if 0 recent accesses).
- **Calculation**: $\frac{N_{\text{reads}}}{N_{\text{reads}} + N_{\text{writes}}}$ (bounded in $[0.0, 1.0]$). Defaults to $0.0$ if no accesses occurred.
- **Directionality**: Higher values mean read-dominated; lower values mean write-dominated.
- **Predictive Value**: Read-heavy workloads benefit disproportionately from SSD promotion because SSD read latency is minimal without endurance write wear.
- **Availability Before $t$**: **Yes**.

---

### Feature 7: `write_ratio`
- **Meaning**: Proportion of operations on the file that were write requests.
- **Observation Window**: $[t - 6\text{h}, t)$ (with fallback to past 12h).
- **Calculation**: $\frac{N_{\text{writes}}}{N_{\text{reads}} + N_{\text{writes}}}$ (bounded in $[0.0, 1.0]$). Defaults to $0.0$ if no accesses occurred.
- **Directionality**: Higher values indicate write-intensive files.
- **Predictive Value**: Storage tiering decision engines often treat writes differently due to SSD flash erase cycles and write amplification factors.
- **Availability Before $t$**: **Yes**.

---

### Feature 8: `average_request_size`
- **Meaning**: Mean I/O payload size in bytes per access.
- **Observation Window**: $[t - 6\text{h}, t)$ (with fallback to past 12h).
- **Calculation**: $\frac{1}{N} \sum \text{request\_size\_bytes}$ (in bytes). Defaults to $0.0$ if unaccessed.
- **Directionality**: Indicates sequential block size vs small random I/O.
- **Predictive Value**: Small random I/O experiences dramatic speedups on SSD compared to HDD, influencing the net performance benefit calculation.
- **Availability Before $t$**: **Yes**.

---

### Feature 9: `current_tier`
- **Meaning**: The physical storage tier where the file currently resides at prediction timestamp $t$.
- **Encoding**:
  - `0`: **HDD** (Hard Disk Drive / High-Capacity Cold Tier)
  - `1`: **SSD** (Solid State Drive / Low-Latency Hot Tier)
- **Calculation**: Most recently recorded tier in trace events for file $f$ prior to $t$. If not yet accessed, defaults to `0` (HDD).
- **Predictive Value**: Necessary context for migration decisions (files on HDD can be promoted; files on SSD can be demoted).
- **Availability Before $t$**: **Yes**.

---

## 3. Target Variable Specification

### Target: `future_access_count`
- **Meaning**: The total number of I/O operations serviced for file $f$ in the upcoming 6-hour prediction window.
- **Temporal Horizon**: $[t, t + 6\text{h})$
- **Data Type**: Non-negative integer (`int64`).
- **Domain**: $[0, 133]$ (Mean = $2.51$, Median = $1.0$, 95th percentile = $11.0$).
- **Role in Pipeline**: The continuous regression target to be predicted by Linear Regression in Phase 5.
- **Data Leakage Rule**: **MUST NEVER** be present in the feature matrix $X$. Used exclusively as target label $y$.

---

## 4. Evaluation-Only Ground Truth

### Field: `eval_true_hotspot_next_6h`
- **Meaning**: Binary ground-truth indicator showing whether file $f$ experiences hotspot conditions during $[t, t + 6\text{h})$.
- **Data Type**: Binary (`0` or `1`).
- **Distribution**: $89.93\%$ Class 0 (Cold/Normal), $10.07\%$ Class 1 (Hotspot).
- **Role in Pipeline**: Strictly isolated from ML model training ($X$ and $y$). Reserved exclusively for evaluating classification metrics (Accuracy, Precision, Recall, F1, False Hotspot Rate) when testing predictions against ground truth.
