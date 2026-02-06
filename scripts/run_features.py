from src.features.build_features import build_features

if __name__ == "__main__":
    X, y, preprocessor = build_features()
    print("Feature engineering completed successfully")
    print(X.head())
