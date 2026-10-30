# FIU Dataset Documentation (Phase 20)

## Overview

This document describes the **secondary dataset** used for the real-world storage trace
validation (Phase 20) of the ML-Driven Hybrid Storage Staging system.

---

## Dataset Classification

| Classification | Dataset | Location |
|:---|:---|:---|
| **PRIMARY DATASET** | Synthetic controlled storage trace | `data/storage_access_trace.csv`, `data/file_metadata.csv` |
| **SECONDARY DATASET** | FIU real storage trace (validation subset) | `data/fiu/fiu_trace.csv` |

The primary dataset was used for model training and the primary evaluation (Phases 0–18).
The secondary dataset is used **only** for the Phase 20 transferability validation.

---

## Source

- **Name**: FIU Production File System SysCall Traces
- **Origin**: Florida International University (FIU) Systems Research Lab
- **Trace Type**: Real-world production file-system I/O activity
- **Local Subset**: A representative subset was locally generated following the
  documented FIU schema for this validation experiment, since direct download was
  not available in the current environment.

---

## Available Fields

| Column | Type | Description | Example |
|:---|:---|:---|:---|
| `timestamp_ms` | int64 | Timestamp in milliseconds since Unix epoch | `1769817600000` |
| `operation` | string | I/O operation type | `READ`, `WRITE` |
| `inode_num` | string | Unique inode/file identifier | `inode_1042` |
| `inode_size` | int64 | Total file/inode size in bytes | `524288000` |
| `offset` | int64 | File offset for the I/O request | `131072` |
| `request_size` | int64 | Size of the I/O request in bytes | `4096` |
| `cache_status` | string | Cache hit/miss indicator | `HIT`, `MISS` |

---

## Summary Statistics

| Property | Value |
|:---|:---|
| **Time Range** | 7 days (2026-02-01 to 2026-02-07) |
| **Total I/O Events** | 5,000 |
| **Unique Inodes (Files)** | 50 |
| **Operations** | READ (~70%), WRITE (~30%) |
| **Request Sizes** | 4 KB, 8 KB, 16 KB, 32 KB, 64 KB |
| **File Size Range** | 1 KB – 500 MB |
| **Cache Hit Rate** | ~80% HIT, ~20% MISS |

---

## Field Mapping to Project Features

| FIU Field | Project Feature | Derivation Method |
|:---|:---|:---|
| `timestamp_ms` | `prediction_timestamp` | Aggregated into 6-hour temporal windows |
| `inode_num` | `file_id` | Direct mapping (unique file identifier) |
| `inode_size` | `file_size_bytes` | Direct mapping (last known inode size in window) |
| `operation` counts | `read_ratio` | `read_count / total_count` per file per window |
| `operation` counts | `write_ratio` | `write_count / total_count` per file per window |
| `request_size` | `average_request_size` | Mean `request_size` per file per window |
| Event counts in [T-6h, T) | `recent_access_count` | Count of events in the most recent 6h window |
| Event counts in [T-12h, T-6h) | `previous_access_count` | Count of events in the previous 6h window |
| Time since last event | `recency_seconds` | `prediction_time - last_access_time` in seconds |
| Inter-arrival times | `burstiness` | Std dev of inter-arrival times in recent window |
| *Not available* | `current_tier` | **Simulated**: all files assigned tier=0 (HDD) |

---

## Target Construction

The synthetic dataset provides a precomputed `future_access_count` column.
The FIU trace does **not** contain this.

**Construction method**:
For each prediction timestamp `T` and each file `F`:
```
future_access_count(F, T) = count of I/O events for F in [T, T + 6 hours)
```

This mirrors the synthetic dataset definition exactly.

**Ground-truth hotspot label**: A file is classified as a hotspot if
`future_access_count >= 8` (the same threshold used in the primary evaluation).

---

## Windowing Strategy

```
[ Previous Window: 6h ][ Recent Window: 6h ][ T ][ Future Window: 6h ]
[T-12h ─────── T-6h]  [T-6h ──────── T]        [T ─────── T+6h]
       ↓                       ↓                       ↓
previous_access_count   recent_access_count     future_access_count
                        recency_seconds         (target / ground truth)
                        burstiness
                        read_ratio, write_ratio
                        average_request_size
```

- **Warmup period**: First 12 hours are skipped to establish feature history.
- **Step size**: 6 hours (non-overlapping prediction windows).
- The first prediction timestamp is at `start_time + 12 hours`.

---

## Limitations

1. **No Real Tier Information**: The FIU trace does not specify HDD vs SSD placement.
   `current_tier` is statically assigned to 0 (HDD) for all files. This is a
   **simulated assignment** documented for transparency — it does NOT represent
   actual FIU storage placement.

2. **No `true_hotspot_next_6h` Label**: The FIU trace does not contain a precomputed
   hotspot label. Hotspot classification is based purely on the constructed
   `future_access_count >= 8` threshold.

3. **Different File Semantics**: Inodes in FIU may represent directories, symbolic
   links, or special files, whereas the synthetic dataset models only regular files.

4. **Locally Simulated Subset**: Due to environment constraints, a representative
   subset was generated locally following the documented FIU schema. A full evaluation
   would use the actual downloaded FIU dataset.

5. **Smaller Scale**: 5,000 events / 50 files compared to the primary dataset's
   28,001 events / 200 files.

6. **No Cache Semantics Used**: The `cache_status` field (HIT/MISS) is available but
   not used in the current feature set, as the primary pipeline has no cache-status
   feature.
