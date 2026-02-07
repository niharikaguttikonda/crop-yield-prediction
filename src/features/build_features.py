print("NEW BUILD_FEATURES FILE IS RUNNING")

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def build_features():
    print("🚀 build_features() function called")

    df = pd.read_csv("data/processed/finaldataset.csv")

    X = df[["Area", "Item", "Year"]]
    y = df["Yield"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Area", "Item"]),
            ("num", "passthrough", ["Year"])
        ]
    )

    return X, y, preprocessor
