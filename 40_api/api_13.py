
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictInput(BaseModel):
    features: list[float]

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Features:", data.feature_names)
print("Features number:", len(data.feature_names))

logreg = LogisticRegression(max_iter=10000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

@app.post("/predict/v1")
def predict_v1(input: PredictInput):
    features = np.array(input.features).reshape(1, -1)
    pred = logreg.predict(features)
    return {"model": "Logistic Regression", "prediction": int(pred[0])}

@app.post("/predict/v2")
def predict_v2(input: PredictInput):
    features = np.array(input.features).reshape(1, -1)
    pred = rf.predict(features)
    return {"model": "Random Forest", "prediction": int(pred[0])}

@app.post("/predict/v3")
def predict_v3(input: PredictInput):
    features = np.array(input.features).reshape(1, -1)
    pred = gb.predict(features)
    return {"model": "Gradient Boosting", "prediction": int(pred[0])}


@app.post("/compare")   
def compare_models(input: PredictInput):
    features = np.array(input.features).reshape(1, -1)
    pred_logreg = logreg.predict(features)[0]
    pred_rf = rf.predict(features)[0]
    pred_gb = gb.predict(features)[0]
    return {
        "Logistic Regression": int(pred_logreg),
        "Random Forest": int(pred_rf),
        "Gradient Boosting": int(pred_gb)
    }
 
# START: uvicorn api_13:app --reload
#  Example JSON to test:
# { "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}
# http://localhost:8000/docs
