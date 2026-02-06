from src.data.preprocess import preprocess_data

if __name__ == "__main__":
    df = preprocess_data()
    print("Preprocessing completed successfully")
    print(df.head())
