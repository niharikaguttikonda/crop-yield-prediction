import pandas as pd
from pathlib import Path

def preprocess_data():
    raw_path = Path("data/raw/yield.csv")
    df = pd.read_csv(raw_path)

    final_df = df[["Area", "Item", "Year", "Value"]].rename(
        columns={"Value": "Yield"}
    )

    output_path = Path("data/processed/finaldataset.csv")
    final_df.to_csv(output_path, index=False)

    return final_df
