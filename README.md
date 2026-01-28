# Time-Series Anomaly Detection Dashboard (Dash)

Dash dashboard for time-series anomaly detection using a rolling z-score anomaly score.

## Tech
Python, Dash, Plotly, Pandas, NumPy

## Features
- Sensor selection (s1/s2/s3)
- Rolling window & threshold controls
- Anomaly markers on the time-series plot
- Top anomalies table + KPI summary

## Run locally
```bash
python -m pip install -r requirements.txt
python src/data_gen.py
python app.py
