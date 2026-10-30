# ML-Driven Hybrid Storage Staging and Hotspot Migration

**A Predictive, Payback-Aware Data Tiering Architecture**

---

## 1. Abstract
This project implements an intelligent storage tiering system that proactively migrates data between a high-capacity, slow tier (simulated HDD) and a low-capacity, fast tier (simulated SSD). By utilizing Machine Learning (Linear Regression) to predict future file access frequencies and applying a strict mathematical payback ratio (Cost-Benefit Analysis), the system minimizes latency while drastically reducing unnecessary data migrations.

## 2. Problem Statement
In hybrid storage environments, relying purely on reactive migration (e.g., LRU or Frequency thresholds) often results in "migration thrashing"—wasting storage bandwidth by moving data after the access burst has already passed, or moving massive files for minimal latency gain.

## 3. Motivation
Modern data centers demand low latency but cannot afford to store all data on premium SSDs. A system that can *predict* hotspots before they occur, and mathematically prove the migration is worth the bandwidth cost, offers a massive efficiency improvement over reactive heuristics.

## 4. Objectives
1. Implement a temporal feature extraction engine to process storage traces without data leakage.
2. Train a Machine Learning model to predict future file access counts.
3. Combine ML predictions with a deterministic Payback Calculator to prevent thrashing.
4. Safely execute and verify simulated physical file migrations using cryptographic checksums.
5. Provide a heuristic feedback loop to self-correct heat scores.

## 5. Proposed Architecture
The architecture functions as a complete pipeline: 
1. **Data Ingestion:** Reads chronological I/O events.
2. **Feature Engineering:** Extracts a 6-hour historical snapshot.
3. **ML Prediction:** Forecasts accesses for the next 6 hours.
4. **Decision Engine:** Evaluates predictions against a Cost-Benefit Payback Ratio.
5. **Execution:** Performs physical `shutil.copy2` migrations with SHA-256 verification.
6. **Feedback:** Adjusts file heat scores based on prediction errors.

## 6. Dataset
**Note:** This project uses a **controlled synthetic dataset** containing 28,000 I/O events across 200 files over 14 days. This allows for rigorous, isolated evaluation of the ML model against a known ground truth. Secondary validation was performed on a real FIU trace, which demonstrated the necessity of domain-specific model retraining (documented in Phase 20).

## 7. Feature Engineering
A sliding window aggregates chronological events into 9 features per file, strictly computed prior to the prediction timestamp. Features include `recent_access_count`, `recency_seconds`, and the Goh-Barabási `burstiness` coefficient.

## 8. ML Model
The project uses Scikit-Learn's `LinearRegression` as a foundational, interpretable baseline. It maps the 9 features to a continuous output: `predicted_future_access_count`.

## 9. Hotspot Detection & Heat-Map
- **Hotspot:** A file predicted to receive ≥ 8 accesses in the next 6 hours.
- **Heat Map:** An independent heuristic score `H = 0.7F + 0.3R` used for ranking and feedback.

## 10. Payback-Aware Migration
The Decision Engine calculates:
- **Cost:** File Size / Migration Bandwidth.
- **Benefit:** Predicted Accesses × (HDD Latency - SSD Latency).
If `Benefit / Cost < 1.0`, the migration is rejected, preventing the movement of massive files for minor gains.

## 11. Simulated HDD/SSD & Feedback
Hardware tiers are simulated using local directories (`storage/HDD` and `storage/SSD`). Latencies (10ms HDD, 0.2ms SSD) are mathematical constants. Post-migration, the feedback engine compares predicted vs. actual accesses and uses a bounded `tanh` function to adjust the file's heat score.

## 12. Baselines & Evaluation
The proposed ML system was evaluated against:
1. **Static Tiering:** No migrations.
2. **Frequency-Based Migration:** Reactive migration based solely on past access count.

**Results:** The ML model reduced average latency by 2.01% compared to static tiering. While the Frequency baseline achieved slightly lower latency, it did so by migrating 24 files (16 unnecessary). The ML system achieved nearly the same benefit using only 5 highly-targeted migrations, reducing bandwidth overhead by 80%.

## 13. Project Limitations
- The primary dataset is synthetic.
- Hardware latency and storage bus bandwidth are mathematically simulated constants.
- Linear Regression is simplistic and achieves moderate recall (48%).
- The feedback loop adjusts heuristic scores but does not retrain the ML weights.

---

## 14. Installation & Execution

### Requirements
- Python 3.9+
- Windows/Linux

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # (Windows)
# source .venv/bin/activate (Linux)
pip install -r requirements.txt
```

### Running the End-to-End Pipeline
To execute data generation, model training, migration simulation, and evaluation from scratch:
```bash
python run_clean_experiment.py
```

### Running the Test Suite
The project includes a robust Pytest suite:
```bash
pytest tests/
```

### Running the Interactive Dashboard
To view results, heat maps, and migration logs:
```bash
streamlit run dashboard/app.py
```

## 15. Project Structure
See `docs/FINAL_PROJECT_STRUCTURE.md` for a detailed breakdown of the codebase architecture. Experimental configuration is centralized in `config.py`.
