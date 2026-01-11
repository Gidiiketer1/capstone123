# src/model.py
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from typing import Tuple

def train_model(model: ClassifierMixin, X: np.ndarray, y: np.ndarray) -> Tuple[ClassifierMixin, float]:
    """
    Train a classifier and return the trained model and test accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, acc
