import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def preprocess_fiu_trace(fiu_csv_path: Path) -> pd.DataFrame:
    """
    Parses the FIU trace, extracts standard ML features, and 
    computes the future access count target.
    
    Returns a DataFrame with columns:
      file_id, prediction_timestamp, recent_access_count, 
      previous_access_count, recency_seconds, burstiness, 
      file_size_bytes, read_ratio, write_ratio, average_request_size, 
      current_tier, future_access_count
    """
    df = pd.read_csv(fiu_csv_path)
    
    # Convert timestamp_ms to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    
    # Map basic fields
    df.rename(columns={"inode_num": "file_id", "inode_size": "file_size_bytes"}, inplace=True)
    
    start_time = df["timestamp"].min()
    end_time = df["timestamp"].max()
    
    # Define observation windows
    warmup_hours = 12
    recent_window_h = 6
    previous_window_h = 6
    future_window_h = 6
    step_h = 6
    
    current_time = start_time + timedelta(hours=warmup_hours)
    
    feature_rows = []
    
    # Iterate over prediction timestamps
    while current_time + timedelta(hours=future_window_h) <= end_time:
        t_pred = current_time
        t_recent_start = t_pred - timedelta(hours=recent_window_h)
        t_prev_start = t_recent_start - timedelta(hours=previous_window_h)
        t_future_end = t_pred + timedelta(hours=future_window_h)
        
        # Split trace into windows
        df_prev = df[(df["timestamp"] >= t_prev_start) & (df["timestamp"] < t_recent_start)]
        df_recent = df[(df["timestamp"] >= t_recent_start) & (df["timestamp"] < t_pred)]
        df_future = df[(df["timestamp"] >= t_pred) & (df["timestamp"] < t_future_end)]
        
        # Consider all files seen so far up to t_pred
        df_history = df[df["timestamp"] < t_pred]
        active_files = df_history["file_id"].unique()
        
        for file_id in active_files:
            file_recent = df_recent[df_recent["file_id"] == file_id]
            file_prev = df_prev[df_prev["file_id"] == file_id]
            file_future = df_future[df_future["file_id"] == file_id]
            
            recent_count = len(file_recent)
            prev_count = len(file_prev)
            future_count = len(file_future)
            
            # Recency
            if recent_count > 0:
                last_access = file_recent["timestamp"].max()
                recency = (t_pred - last_access).total_seconds()
            else:
                # Check older history
                file_hist = df_history[df_history["file_id"] == file_id]
                if len(file_hist) > 0:
                    last_access = file_hist["timestamp"].max()
                    recency = (t_pred - last_access).total_seconds()
                else:
                    recency = config.MAX_RECENCY_SECONDS
            
            # Burstiness
            if recent_count > 1:
                times = file_recent["timestamp"].sort_values().values
                diffs = (times[1:] - times[:-1]) / np.timedelta64(1, 's')
                burstiness = float(np.std(diffs))
            else:
                burstiness = -1.0
                
            # Ratios and sizes (from recent window, fallback to history)
            if recent_count > 0:
                read_count = (file_recent["operation"] == "READ").sum()
                write_count = (file_recent["operation"] == "WRITE").sum()
                read_ratio = read_count / recent_count
                write_ratio = write_count / recent_count
                avg_req_size = file_recent["request_size"].mean()
                file_size = file_recent["file_size_bytes"].iloc[-1]
            else:
                read_ratio = 0.0
                write_ratio = 0.0
                avg_req_size = 0.0
                file_hist = df_history[df_history["file_id"] == file_id]
                file_size = file_hist["file_size_bytes"].iloc[-1]
                
            # Current Tier: Not available in FIU. Assign to HDD (0) as default for evaluation.
            current_tier = config.TIER_HDD_ENCODING
            
            feature_rows.append({
                "file_id": file_id,
                "prediction_timestamp": t_pred,
                "recent_access_count": recent_count,
                "previous_access_count": prev_count,
                "recency_seconds": recency,
                "burstiness": burstiness,
                "file_size_bytes": file_size,
                "read_ratio": read_ratio,
                "write_ratio": write_ratio,
                "average_request_size": avg_req_size,
                "current_tier": current_tier,
                "future_access_count": future_count
            })
            
        current_time += timedelta(hours=step_h)
        
    return pd.DataFrame(feature_rows)

if __name__ == "__main__":
    fiu_path = Path("data/fiu/fiu_trace.csv")
    if not fiu_path.exists():
        print("Run generate_fiu_mock.py first!")
        sys.exit(1)
        
    print("Preprocessing FIU trace...")
    features_df = preprocess_fiu_trace(fiu_path)
    print(f"Extracted {len(features_df)} feature vectors.")
    
    out_path = Path("data/fiu/fiu_features.csv")
    features_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
