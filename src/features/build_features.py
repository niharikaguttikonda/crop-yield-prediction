import pandas as pd
from sklearn.preprocessing import LabelEncoder

def build_features():
    df = pd.read_csv("data/processed/finaldataset.csv")

    le_area = LabelEncoder()
    le_item = LabelEncoder()

    df["Area_enc"] = le_area.fit_transform(df["Area"])
    df["Item_enc"] = le_item.fit_transform(df["Item"])

    X = df[["Area_enc", "Item_enc", "Year"]]
    y = df["Yield"]

    return X, y
