from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load Iris dataset and train a simple model at startup
iris = load_iris()
X, y = iris.data, iris.target
model = LogisticRegression(max_iter=200)
model.fit(X, y)

app = FastAPI(title="Iris Prediction API", description="Predict Iris species from sepal and petal measurements.", version="0.1")

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict_species(request: IrisRequest):
    # Prepare input array for prediction
    input_array = np.array([[request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]])
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array).max()
    return {
        "prediction": iris.target_names[prediction],
        "probability": float(probability)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
