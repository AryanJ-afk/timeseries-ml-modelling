import pandas as pd
import numpy as np

INPUT_PATH = "test.csv"
OUTPUT_PATH = "driftedtest.csv"

np.random.seed(42)

df = pd.read_csv(INPUT_PATH)

for col in ["lag_1", "lag_2", "lag_24", "lag_48", "rolling_mean_24"]:
    df[col] = df[col] * 1.35 + np.random.normal(0, 0.08, len(df))

df["rolling_std_24"] = df["rolling_std_24"] * 1.5 + np.random.normal(0, 0.03, len(df))

# Push temporal pattern slightly
df["hour"] = (df["hour"] + 3) % 24

df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved drifted CSV to {OUTPUT_PATH}")
