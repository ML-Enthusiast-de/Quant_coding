# src/models_tree.py
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def train_tree_regressor(
    X_train,
    y_train,
    max_depth: int = 3,
    max_iter: int = 300,
    learning_rate: float = 0.05,
    l2_regularization: float = 0.0,
):
    """
    Train a HistGradientBoostingRegressor.
    Defaults reproduce your previous baseline.
    """
    model = HistGradientBoostingRegressor(
        max_depth=max_depth,
        max_iter=max_iter,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_regression(
    model: HistGradientBoostingRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return {"mse": mse, "preds": preds}
