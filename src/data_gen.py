import numpy as np
import pandas as pd
from pathlib import Path

def generate_sensor_series(n=6000, seed=42, sensor_name="s1"):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=n, freq="s")


    base = 0.6 * np.sin(np.linspace(0, 20*np.pi, n)) + 0.05 * rng.normal(size=n)

    drift = np.zeros(n)
    drift_start = int(n * 0.55)
    drift[drift_start:] = np.linspace(0, 1.2, n - drift_start)

    x = base + drift

    spike_idx = rng.choice(np.arange(200, n-200), size=18, replace=False)
    x[spike_idx] += rng.normal(loc=2.5, scale=0.6, size=len(spike_idx))

    drop_idx = rng.choice(np.arange(200, n-200), size=10, replace=False)
    x[drop_idx] -= rng.normal(loc=2.2, scale=0.5, size=len(drop_idx))

    for _ in range(3):
        start = int(rng.integers(300, n-400))
        length = int(rng.integers(40, 120))
        x[start:start+length] = x[start] + 0.005 * rng.normal(size=length)

    return pd.DataFrame({"timestamp": t, "sensor": sensor_name, "value": x})

def main():
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    df1 = generate_sensor_series(seed=42, sensor_name="s1")
    df2 = generate_sensor_series(seed=99, sensor_name="s2")
    df3 = generate_sensor_series(seed=202, sensor_name="s3")

    df2["value"] = 0.8 * df2["value"] + 0.2
    rng = np.random.default_rng(7)
    df3["value"] = df3["value"] + 0.12 * rng.normal(size=len(df3))

    df = pd.concat([df1, df2, df3], ignore_index=True)
    df.to_csv(out_dir / "sensor_data.csv", index=False)

    print("ok Generated: data/sensor_data.csv")
    print(df.head())

if __name__ == "__main__":
    main()
