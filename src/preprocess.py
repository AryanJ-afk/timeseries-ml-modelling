import pandas as pd
from load_data import load_dataset
import os


def create_hourly_series(path):
    df = load_dataset(path)

    demand = df["Global_active_power"]

    # fill small gaps
    demand = demand.ffill()

    # convert minute data to hourly average
    hourly = demand.resample("h").mean()

    return hourly


if __name__ == "__main__":
    path = "data/raw/household_power_consumption.txt"

    hourly = create_hourly_series(path)

    print(hourly.head())
    print(hourly.describe())
    os.makedirs("data/processed", exist_ok=True)
    hourly.to_csv("data/processed/hourly_demand.csv")