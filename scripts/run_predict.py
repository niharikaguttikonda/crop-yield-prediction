from src.inference.predict import predict_yield

if __name__ == "__main__":
    result = predict_yield(
        area="India",
        item="Rice",
        year=2025
    )

    print("🌾 Predicted Crop Yield:", round(result, 2))

