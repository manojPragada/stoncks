from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from stoncks_app.data import Dataset


@dataclass
class ModelResult:
    model: RandomForestClassifier
    report: str


def train_model(dataset: Dataset) -> ModelResult:
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.features, dataset.labels, test_size=0.2, shuffle=False
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    report = classification_report(y_test, preds)
    return ModelResult(model=model, report=report)


def save_model(model: RandomForestClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> RandomForestClassifier:
    return joblib.load(path)


def predict_proba(model: RandomForestClassifier, features: pd.DataFrame) -> float:
    proba = model.predict_proba(features)[0]
    return float(proba[1])
