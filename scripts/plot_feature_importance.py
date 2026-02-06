import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load trained pipeline
model = joblib.load("models/crop_yield_model.pkl")

# Extract trained RandomForest
rf = model.named_steps["regressor"]

# Get feature names after OneHotEncoding
preprocessor = model.named_steps["preprocessor"]
feature_names = preprocessor.get_feature_names_out()

# Create importance dataframe
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False).head(15)

# Plot
plt.figure(figsize=(10,6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.title("Top Feature Importances for Crop Yield Prediction")
plt.gca().invert_yaxis()
plt.show()
