import pandas as pd

# Read the CSV
df = pd.read_csv("test.csv")

# Remove the first 2 columns
df = df.iloc[:, 2:]

# Save back to the same CSV
df.to_csv("test.csv", index=False)
