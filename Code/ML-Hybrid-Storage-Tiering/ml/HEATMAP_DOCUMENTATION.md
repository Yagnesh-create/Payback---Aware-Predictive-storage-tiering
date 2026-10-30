# Heat-Map Engine Documentation

## 1. Purpose

The heat-map engine quantifies how "hot" each file currently is based on its
**historical** access behavior. It does NOT predict future behavior — that is
the ML model's job. The heat score provides a human-interpretable summary of
recent file activity that feeds into the integrated decision pipeline.

---

## 2. Heat Score Formula

```
H = α × frequency_score + β × recency_score
```

| Symbol | Name | Default | Source |
| :----- | :--- | :------ | :----- |
| H | Heat score | [0, 1] | Composite output |
| α | Frequency weight | **0.7** | `config.HEAT_FREQUENCY_WEIGHT` |
| β | Recency weight | **0.3** | `config.HEAT_RECENCY_WEIGHT` |

> **IMPORTANT**: α = 0.7 and β = 0.3 are **initial configuration choices**,
> NOT experimentally proven optimal values. They can be changed in `config.py`
> without modifying the algorithm implementation.

---

## 3. Frequency Score

**Input**: `recent_access_count` — number of I/O accesses in the most recent
6-hour observation window `[t-6h, t)`.

**Normalization**: Min-max normalization across all files at the current
evaluation point:

```
frequency_score = (recent_access_count - min) / (max - min)
```

- Range: [0.0, 1.0]
- If all files have the same access count (`max == min`), all scores are 0.0.
- Higher access count → higher frequency score → hotter.

---

## 4. Recency Score

**Input**: `recency_seconds` — seconds since the file was last accessed.

**Key semantic issue**: Smaller `recency_seconds` = more recently accessed.
This is the *opposite* of "hotness", so we must **invert** the normalization.

**Method**:

```
recency_score = 1.0 - min_max_normalize(recency_seconds)
```

Step by step:
1. Apply min-max normalization: `norm = (recency_seconds - min) / (max - min)`
   - The file with the smallest `recency_seconds` (most recent) gets `norm = 0.0`
   - The file with the largest `recency_seconds` (least recent) gets `norm = 1.0`
2. Invert: `recency_score = 1.0 - norm`
   - Most recently accessed → recency_score = **1.0** (hot)
   - Least recently accessed → recency_score = **0.0** (cold)

- Range: [0.0, 1.0]
- If all files have the same recency (equal), all get score 1.0 (all equally "recent").

---

## 5. Interpretation

| Heat Score | Interpretation |
| :--------- | :------------- |
| > 0.8 | Very hot — frequently and recently accessed |
| 0.5 – 0.8 | Warm — moderate activity |
| 0.2 – 0.5 | Cool — low recent activity |
| < 0.2 | Cold — rarely or long-ago accessed |

---

## 6. Output Columns

The heat-map engine adds these columns to the input DataFrame:

| Column | Type | Range | Description |
| :----- | :--- | :---- | :---------- |
| `frequency_score` | float | [0, 1] | Min-max normalized access frequency |
| `recency_score` | float | [0, 1] | Inverted min-max normalized recency |
| `heat_score` | float | [0, 1] | Composite H = α·freq + β·rec |

---

## 7. Limitations

1. **Not a prediction**: Heat score summarizes *past* behavior. It does not
   predict future hotspot status. The trained ML model serves that purpose.
2. **Relative scoring**: Min-max normalization makes scores relative to the
   current file population. Adding or removing files changes all scores.
3. **Single-window view**: Only the most recent 6-hour window contributes
   to the frequency component. Longer-term trends are not captured.
4. **Weight sensitivity**: The 0.7/0.3 split prioritizes frequency over
   recency. Different workloads may benefit from different weights.
5. **Constant-population edge case**: When all files have identical access
   counts or recency, normalization collapses to uniform scores.

---

## 8. File Location

- Implementation: `ml/heatmap.py`
- Configuration: `config.py` → `HEAT_FREQUENCY_WEIGHT`, `HEAT_RECENCY_WEIGHT`
