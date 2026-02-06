from src.inference.predict import predict_yield

if __name__ == "__main__":
    result = predict_yield(10, 5, 2025)
    print("Predicted Yield:", result)
