# src/utility.py
import joblib
import numpy as np
from typing import Any, Dict, List, Union

def save_model(model: Any, scaler: Any, encoders: Dict[str, Any], path: str = "model.joblib") -> None:
    joblib.dump({"model": model, "scaler": scaler, "encoders": encoders}, path)
    print(f"Model saved to {path}")

def load_model(path: str = "model.joblib") -> Dict[str, Any]:
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"No model file found at {path}")

def prepare_input(values: Union[List[Any], np.ndarray]) -> np.ndarray:
    return np.array([values])
