# src/predict.py
import joblib
from sklearn.datasets import load_iris

MODEL_PATH = "models/iris_model.pkl"

def load_model(path=MODEL_PATH):
    """Load and return the trained model."""
    model = joblib.load(path)
    return model

def predict_sample(model, sample):
    """sample: list or array of 4 features -> returns predicted class name."""
    iris = load_iris()
    pred_idx = model.predict([sample])[0]
    pred_name = iris.target_names[pred_idx]
    return int(pred_idx), pred_name

if __name__ == "__main__":
    # Example usage: use the first iris row
    iris = load_iris()
    sample = iris.data[0].tolist()
    model = load_model()
    idx, name = predict_sample(model, sample)
    print("Sample:", sample)
    print("Predicted class index:", idx)
    print("Predicted class name:", name)
