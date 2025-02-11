import pytest
import numpy as np
from src.Lab_2_4_LR2 import (
    LinearRegressor,
    evaluate_regression,
    one_hot_encode,
)  # Assuming the class is in linear_regressor.py


def test_evaluate_regression():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    results = evaluate_regression(y_true, y_pred)

    expected_r2 = 0.9486081370449679  # Precomputed expected value
    expected_rmse = 0.612372  # Precomputed expected value
    expected_mae = 0.5  # Precomputed expected value

    assert np.isclose(results["R2"], expected_r2, atol=1e-6)
    assert np.isclose(results["RMSE"], expected_rmse, atol=1e-6)
    assert np.isclose(results["MAE"], expected_mae, atol=1e-6)


def test_one_hot_encode():
    X = np.array(
        [
            ["Red", 10],
            ["Blue", 20],
            ["Green", 30],
            ["Red", 40],
        ],
        dtype=object,
    )
    categorical_indices = [0]
    transformed_X = one_hot_encode(X, categorical_indices, drop_first=False)

    assert (
        transformed_X.shape[1] == 4
    )  # Original numeric column + 3 one-hot encoded columns

    assert all(transformed_X[:, :-1].sum(axis=1) == 1.0), "All rows must sum to one"

    transformed_X = one_hot_encode(X, categorical_indices, drop_first=True)
    assert (
        transformed_X.shape[1] == 3
    )  # Original numeric column + 3 one-hot encoded columns


def test_fit_gradient_descent():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    model = LinearRegressor()
    model.fit(X, y, method="gradient_descent", learning_rate=0.01, iterations=10000)

    assert np.isclose(model.intercept, 0, atol=1e-1)
    assert np.isclose(model.coefficients[0], 2, atol=1e-1)
