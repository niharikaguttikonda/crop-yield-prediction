import pandas as pd
import joblib

def predict_yield(area: str, item: str, year: int):
    """
    Predict crop yield using trained pipeline
    """
    model = joblib.load("models/crop_yield_model.pkl")

    input_df = pd.DataFrame([{
        "Area": area,
        "Item": item,
        "Year": year
    }])

    prediction = model.predict(input_df)[0]
    return prediction
