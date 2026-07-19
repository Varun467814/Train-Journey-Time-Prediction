import joblib
from pathlib import Path

# Prediction module

def predict(model, features):
    """Make predictions using the trained model.

    Args:
        model: A trained scikit-learn estimator.
        features: Feature matrix as a pandas DataFrame or numpy array.

    Returns:
        Predicted values from the model.
    """
    return model.predict(features)


def load_model(path):
    """Load a trained model from disk."""
    path = Path(path)
    return joblib.load(path)