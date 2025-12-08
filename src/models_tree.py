# src/models_tree.py
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def train_tree_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> HistGradientBoostingRegressor:
    """
    Simple boosted tree baseline that predicts next-day returns.
    """
    model = HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=0.0,
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
