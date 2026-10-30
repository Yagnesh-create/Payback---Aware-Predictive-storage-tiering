# Project Specification: ML-Driven Hybrid Storage Staging and Hotspot Migration

## 1. Project Title & Overview
- **Title**: ML-Driven Hybrid Storage Staging and Hotspot Migration
- **Domain**: Operating Systems / Storage Systems & Machine Learning
- **Context**: College OS Lab Project — designed to be clean, reproducible, explainable, and defendable in a viva examination.

---

## 2. Project Objective
Build an end-to-end, ML-driven hybrid storage tiering system that:
1. Analyzes historical storage-access trace logs.
2. Extracts meaningful temporal and structural I/O features.
3. Trains a genuine Machine Learning model to forecast future access intensity.
4. Identifies prospective file "hotspots" from predicted access counts via a configurable threshold.
5. Evaluates a cost-benefit / payback model before deciding whether to migrate a file between tiers.
6. Simulates multi-tiered physical storage (`storage/HDD/` and `storage/SSD/`) with real file movements.
7. Observes post-migration behavior and incorporates feedback into subsequent scheduling decisions.
8. Evaluates system performance improvements against standard baseline storage policies.

---

## 3. End-to-End System Pipeline
The workflow follows a 13-stage sequential architecture:
```
[1. Dataset]
      │
      ▼
[2. Data Preprocessing]
      │
      ▼
[3. Feature Engineering]
      │
      ▼
[4. ML Model Training] (Linear Regression)
      │
      ▼
[5. Future Access Prediction]
      │
      ▼
[6. Heat-Map Generation]
      │
      ▼
[7. Hotspot Detection] (Threshold-based classification)
      │
      ▼
[8. Migration Cost / Benefit Calculation]
      │
      ▼
[9. Migration Decision Engine] (Payback & Utility check)
      │
      ▼
[10. HDD / SSD Physical File Migration]
      │
      ▼
[11. Post-Migration Monitoring]
      │
      ▼
[12. Feedback Loop]
      │
      ▼
[13. Performance Evaluation & Baseline Comparison]
```

---

## 4. Machine Learning Formulation

### 4.1 Primary ML Model
- **Algorithm**: **Linear Regression** (`sklearn.linear_model.LinearRegression`).
- **Training Requirement**: Fully trained on historical data using genuine `model.fit()` with proper train/test splits. No mock predictions or hardcoded placeholders.
- **Role**: Predict continuous **future access count** for each file over the prediction horizon.
- **Hotspot Classification**: Derived downstream from predicted access counts using a configurable threshold $\theta_{\text{hotspot}}$:
  $$\text{Is\_Hotspot} = \begin{cases} 1 & \text{if } \widehat{\text{Future\_Accesses}} \ge \theta_{\text{hotspot}} \\ 0 & \text{otherwise} \end{cases}$$

### 4.2 Feature Set
Engineered strictly from historical observation windows and static metadata:
1. **Recent Access Frequency**: Number of I/O operations in the current observation window.
2. **Previous-Window Access Frequency**: Number of I/O operations in the preceding observation window.
3. **Recency**: Elapsed time since the last access to the file.
4. **Burstiness**: Variance / standard deviation of inter-arrival access times within the window.
5. **File Size**: Size of the file in bytes/KB from metadata.
6. **Read/Write Ratio**: Ratio of read operations to write operations for the file.
7. **Current Storage Tier**: Binary/encoded indicator (0 for HDD, 1 for SSD).

### 4.3 Target Variable
- **Target ($y$)**: `future_access_count` (total number of accesses observed in the subsequent prediction window).

### 4.4 Data-Leakage Rule
- **CRITICAL RESTRICTION**: The dataset column `true_hotspot_next_6h` represents ground-truth future label data.
- **Rule**: `true_hotspot_next_6h` **MUST NEVER** be passed to the model as an input feature during preprocessing or training.
- **Permitted Use**: Exclusively reserved for evaluating downstream classification accuracy, precision, recall, and false-hotspot rate during the validation/testing phase.

---

## 5. Storage Tier Simulation Model
- **Tiers**:
  - `storage/HDD/`: Simulates high-capacity, higher-latency primary/cold tier.
  - `storage/SSD/`: Simulates low-capacity, low-latency performance/hot tier.
- **Mechanism**: Directory-based local storage. The system must physically move or copy real test files between these directories during migration execution.
- **Latency & Capacity Models**:
  - Simulated latency parameters calibrated to realistic HDD vs. SSD read/write profiles.
  - SSD capacity limit enforcing eviction or promotion constraints.

---

## 6. Migration Decision Logic
A predicted hotspot does **not** trigger an automatic migration. A migration is only scheduled if the operational benefit outweighs the overhead.

The decision engine balances:
1. **Predicted Future Accesses ($\widehat{A}_i$)**: Frequency expected in the upcoming period.
2. **Hotspot Classification**: Verification that $\widehat{A}_i \ge \theta_{\text{hotspot}}$.
3. **Estimated Migration Cost ($C_{\text{mig}}$)**:
   - Function of file size, bus transfer bandwidth, and I/O penalty imposed on running workloads.
4. **Expected Performance Benefit ($B_{\text{perf}}$)**:
   - Expected I/O latency reduction from servicing $\widehat{A}_i$ accesses from SSD instead of HDD:
     $$B_{\text{perf}} \approx \widehat{A}_i \times (\text{Latency}_{\text{HDD}} - \text{Latency}_{\text{SSD}})$$
5. **Net Payback / Utility ($U$)**:
   - Migration executed only if $B_{\text{perf}} > C_{\text{mig}} + \text{Threshold}_{\text{margin}}$.
   - Downward demotion (SSD $\to$ HDD) triggered when files cool down to reclaim SSD capacity.

---

## 7. Comparative Baselines
To validate the ML-driven approach, the framework will implement and compare against three policies:
1. **Static Tiering**: Files remain on their initially assigned tier; no dynamic migration occurs.
2. **Frequency-Based Reactive Migration**: Migration triggered strictly by past access counters exceeding a static threshold (traditional reactive cache/tiering policy).
3. **Proposed ML-Driven Migration**: Predictive, cost-benefit-aware migration governed by the trained regression model and payback utility engine.

---

## 8. Evaluation Framework

### 8.1 Machine Learning Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Coefficient of Determination ($R^2$)**
- **Classification Metrics (derived hotspot vs. `true_hotspot_next_6h`)**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

### 8.2 Systems & Storage Metrics
- **Average Response Time / Latency** (ms per I/O operation)
- **Overall System Throughput** (MB/s or IOPS)
- **SSD Capacity Utilization** (%)
- **Total Migration Count** (promotions and demotions)
- **Migration Overhead** (I/O and time spent migrating data)
- **Migration Success Rate / Hit Rate** (percentage of promoted files that were actually accessed heavily in the target window)
- **False Hotspot Rate** (files unnecessarily promoted to SSD)

---

## 9. Constraints & Viva Guidelines
- **Target Environment**: Standard Python stack (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`).
- **All code location**: All implementation scripts, modules, and tests must reside inside the [`Code/`](file:///d:/THEJA%20VIT/OS%20operating%20system/Lab%20Project/Code) directory.
- **Dataset location**: Loaded from [`DataBase/`](file:///d:/THEJA%20VIT/OS%20operating%20system/Lab%20Project/DataBase).
- **Simplicity & Explainability**: No unnecessary complex architectures (e.g., GNNs, Federated Learning, Reinforcement Learning, Causal AI, Conformal Prediction, or Digital Twins). Every equation, feature, and decision step must be straightforward to explain and defend in front of an academic evaluator.
