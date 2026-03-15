import pandas as pd


def load_dataset(path):
    df = pd.read_csv(
        path,
        sep=';',
        na_values=['?'],
        low_memory=False
    )

    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce')
    df = df.drop(columns=['Date', 'Time'])
    df = df.set_index('datetime')

    cols = df.columns
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    return df


if __name__ == "__main__":
    path = "data/raw/household_power_consumption.txt"

    df = load_dataset(path)

    print(df.head())
    print(df.info())