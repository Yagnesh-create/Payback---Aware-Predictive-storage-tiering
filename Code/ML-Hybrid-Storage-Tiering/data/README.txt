ML-Driven Hybrid Storage Staging and Hotspot Migration Dataset
================================================================

storage_access_trace.csv
- 14 days of synthetic, trace-like I/O events
- 200 files
- READ/WRITE operations
- timestamps, file sizes, offsets, request sizes
- simulated SSD/HDD tier and response time
- true_hotspot_next_6h is future ground truth and MUST NOT be used as an input

file_metadata.csv
- static file metadata

Suggested model inputs
----------------------
Recent access frequency
Previous-window access frequency
Recency
Mean inter-access time
Burstiness
File size
Read/write ratio
Current tier

Target
------
Future access count OR true_hotspot_next_6h

Suggested evaluation
--------------------
Accuracy, precision, recall, F1, false-hotspot rate,
migration count, migration overhead, average latency,
SSD utilization, and throughput.

The dataset is custom-created with random seed 42 for reproducibility.
