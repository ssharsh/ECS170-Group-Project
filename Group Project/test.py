from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent


def inspect_dataset(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path}. Put {name} in the same folder as this script."
        )

    data = pd.read_csv(path, header=None)
    print(f"\n{name}")
    print(f"Rows: {len(data):,} | Columns: {data.shape[1]:,}")
    print("\nFirst five rows:")
    print(data.head().to_string(index=False))
    print("\nColumn data types:")
    print(data.dtypes.to_string())
    print(f"\nMissing values: {int(data.isna().sum().sum()):,}")

    if data.shape[1] > 0:
        print("\nValues in the first column (assuming it contains the class label):")
        print(data.iloc[:, 0].value_counts(dropna=False).sort_index().to_string())

    numeric = data.select_dtypes(include="number")
    if not numeric.empty:
        print("\nNumeric column summary:")
        print(numeric.describe().T.to_string())

    return data


def main() -> None:
    train = inspect_dataset("train.csv")
    test = inspect_dataset("test.csv")
    if train.shape[1] != test.shape[1]:
        print(
            "\nWARNING: train and test have different numbers of columns "
            f"({train.shape[1]} vs {test.shape[1]})."
        )


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        print(error)
        raise SystemExit(1) from error