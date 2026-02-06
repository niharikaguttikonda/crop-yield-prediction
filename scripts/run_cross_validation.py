from sklearn.model_selection import cross_val_score
from src.features.build_features import build_features
from src.models.train_model import train_model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

X, y, preprocessor = build_features()

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ))
])

scores = cross_val_score(model, X, y, cv=5, scoring="r2")

print("Cross-validation R² scores:", scores)
print("Mean R²:", scores.mean())
