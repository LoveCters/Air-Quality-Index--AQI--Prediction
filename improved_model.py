"""Improved AQI prediction using plain Python linear regression.

This script trains a multiple linear regression model using gradient
-descent with only the Python standard library. It avoids external
dependencies while providing better evaluation metrics compared to the
baseline notebook model.
"""

from __future__ import annotations

import csv
import random
import math
from typing import List, Tuple

# type aliases
Vector = List[float]
Matrix = List[Vector]


def load_dataset(path: str) -> List[Tuple[Vector, float]]:
    """Load the CSV dataset and return feature/target pairs.

    Rows missing any required value are skipped. Only the pollutant
    columns used in the original notebook are kept.
    """
    rows: List[Tuple[Vector, float]] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features = [
                    float(row[name])
                    for name in ["CO", "Ozone", "PM10", "PM25", "NO2"]
                ]
                target = float(row["Overall AQI Value"])
            except (ValueError, KeyError):
                # Skip rows with missing or non-numeric data
                continue
            rows.append((features, target))
    return rows


def train_linear_regression(
    data: List[Tuple[Vector, float]],
    *,
    epochs: int = 200,
    learning_rate: float = 1e-5,
    seed: int = 42,
) -> Tuple[Vector, Vector, Vector, Vector, Vector]:
    """Train a simple linear regression model using SGD.

    Returns the model coefficients and the train/test split used for
    evaluation.
    """
    random.seed(seed)
    random.shuffle(data)
    features = [row[0] for row in data]
    targets = [row[1] for row in data]
    n = len(features)
    split = int(0.8 * n)
    X_train, X_test = features[:split], features[split:]
    y_train, y_test = targets[:split], targets[split:]

    n_features = len(features[0])
    beta = [0.0] * (n_features + 1)  # intercept + weights

    for _ in range(epochs):
        for x_vec, y_val in zip(X_train, y_train):
            pred = beta[0] + sum(b * x for b, x in zip(beta[1:], x_vec))
            err = pred - y_val
            beta[0] -= learning_rate * err
            for j in range(n_features):
                beta[j + 1] -= learning_rate * err * x_vec[j]

    return beta, X_train, y_train, X_test, y_test


def predict(beta: Vector, x_vec: Vector) -> float:
    return beta[0] + sum(b * x for b, x in zip(beta[1:], x_vec))


def evaluate(beta: Vector, X: List[Vector], y: List[float]) -> None:
    preds = [predict(beta, x) for x in X]
    n = len(y)
    mae = sum(abs(p - t) for p, t in zip(preds, y)) / n
    mse = sum((p - t) ** 2 for p, t in zip(preds, y)) / n
    rmse = math.sqrt(mse)
    mean_y = sum(y) / n
    ss_tot = sum((t - mean_y) ** 2 for t in y)
    ss_res = sum((p - t) ** 2 for p, t in zip(preds, y))
    r2 = 1 - ss_res / ss_tot
    mape = sum(abs((t - p) / t) for p, t in zip(preds, y)) / n * 100

    print("=== IMPROVED MODEL EVALUATION ===")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


def main() -> None:
    data = load_dataset("AirQualityIndex6years.csv")
    beta, _, _, X_test, y_test = train_linear_regression(data)
    evaluate(beta, X_test, y_test)


if __name__ == "__main__":
    main()
