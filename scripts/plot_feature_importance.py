import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load trained model
model = joblib.load("models/crop_yield_model.pkl")

# Get feature importances
importances = model.feature_importances_

# Feature names (based on training pipeline)
feature_names = [
    "Item_Potatoes",
    "Year",
    "Item_Sweet potatoes",
    "Item_Cassava",
    "Item_Yams",
    "Other Items",
    "Area_Israel",
    "Area_United States",
    "Area_Netherlands",
    "Area_New Zealand",
    "Area_Japan",
    "Area_Belgium-Luxembourg",
    "Area_Switzerland",
    "Area_United Kingdom",
    "Area_Cook Islands"
]

# Sort features by importance
indices = np.argsort(importances)
sorted_importances = importances[indices]
sorted_features = np.array(feature_names)[indices]

# Plot
plt.figure(figsize=(10, 7))
plt.barh(sorted_features, sorted_importances)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance for Crop Yield Prediction")

# Fix label visibility (macOS)
plt.tight_layout()
plt.subplots_adjust(left=0.35)

# Save plot
plt.savefig("outputs/feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()
