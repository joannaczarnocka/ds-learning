import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class PredictInput(BaseModel):
    features: list[float | None]

data = load_breast_cancer()
X = data.data
y = data.target

n_features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_imputed)

feature_medians = np.median(X_train, axis=0)
feature_stds = np.std(X_train, axis=0)

models = {
    "v1": LogisticRegression(max_iter=1000),
    "v2": RandomForestClassifier(random_state=42),
    "v3": GradientBoostingClassifier(random_state=42)
}

for model in models.values():
    model.fit(X_train_scaled, y_train)

def preprocess_features(features: list[float | None]):
    if len(features) != n_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {n_features} features, got {len(features)}"
        )

    x = np.array(features, dtype=float).reshape(1, -1)
    x = imputer.transform(x)

    lower_bounds = feature_medians - 3 * feature_stds
    upper_bounds = feature_medians + 3 * feature_stds

    outlier_mask = (x[0] < lower_bounds) | (x[0] > upper_bounds)

    if np.any(outlier_mask):
        outlier_indices = np.where(outlier_mask)[0].tolist()
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Outlier detected",
                "outlier_feature_indices": outlier_indices
            }
        )

    x_scaled = scaler.transform(x)

    return x_scaled


@app.post("/predict/{version}")
def predict(version: str, input: PredictInput):
    if version not in models:
        raise HTTPException(status_code=404, detail="Model version not found")

    model = models[version]

    features = preprocess_features(input.features)

    prediction = model.predict(features)
    probability = model.predict_proba(features).tolist()[0]

    return {
        "model_version": version,
        "prediction": int(prediction[0]),
        "probability": probability
    }
    
# START: uvicorn api_13_17:app --reload
#  Example JSON to test (outlier):
# { "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}
# Example JSON to test (not outlier):
# {
#   "features": [
#     14.69, 13.98, 98.22, 656.1, 0.1031,
#     0.1836, 0.1450, 0.0630, 0.2086, 0.0741,
#     0.5462, 1.5110, 4.7950, 49.45, 0.009976,
#     0.05244, 0.05278, 0.01580, 0.02653, 0.005444,
#     16.46, 18.34, 114.10, 809.2, 0.1312,
#     0.3635, 0.3219, 0.1108, 0.2827, 0.09208
#   ]
# }
# http://localhost:8000/docs