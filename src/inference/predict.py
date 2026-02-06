import pandas as pd
import joblib

def predict_yield(area_enc, item_enc, year):
    model = joblib.load("models/crop_yield_model.pkl")

    df = pd.DataFrame(
        [[area_enc, item_enc, year]],
        columns=["Area_enc", "Item_enc", "Year"]
    )

    return model.predict(df)[0]
