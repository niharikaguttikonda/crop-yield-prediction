from src.features.build_features import build_features
from src.models.train_model import train_model

if __name__ == "__main__":
    X, y = build_features()
    model, mse, r2 = train_model(X, y)

    print("Model trained successfully")
    print("MSE:", mse)
    print("R2 Score:", r2)
