from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    features: pd.DataFrame
    labels: pd.Series


def load_price_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {', '.join(sorted(missing))}")
    data = data.sort_values("timestamp")
    return data


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_support_resistance(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    support = data["low"].rolling(window).min()
    resistance = data["high"].rolling(window).max()
    return pd.DataFrame({"support": support, "resistance": resistance})


def compute_momentum(data: pd.DataFrame, window: int = 10) -> pd.Series:
    return data["close"].diff(window)


def compute_breakout_flags(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    rolling_high = data["high"].rolling(window).max()
    rolling_low = data["low"].rolling(window).min()
    breakout_up = (data["close"] > rolling_high.shift(1)).astype(int)
    breakout_down = (data["close"] < rolling_low.shift(1)).astype(int)
    return pd.DataFrame({"breakout_up": breakout_up, "breakout_down": breakout_down})


def compute_fvg_flags(data: pd.DataFrame) -> pd.DataFrame:
    gap_up = (data["low"] > data["high"].shift(1)).astype(int)
    gap_down = (data["high"] < data["low"].shift(1)).astype(int)
    return pd.DataFrame({"fvg_up": gap_up, "fvg_down": gap_down})


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)
    features["return"] = data["close"].pct_change()
    features["rsi"] = compute_rsi(data["close"])
    features["volume"] = data["volume"]
    support_res = compute_support_resistance(data)
    features = pd.concat([features, support_res], axis=1)
    features["momentum"] = compute_momentum(data)
    features = pd.concat([features, compute_breakout_flags(data)], axis=1)
    features = pd.concat([features, compute_fvg_flags(data)], axis=1)
    return features.fillna(0)


def build_labels(data: pd.DataFrame, horizon: int = 3) -> pd.Series:
    future_return = data["close"].shift(-horizon) / data["close"] - 1
    return (future_return > 0).astype(int)


def make_dataset(path: Path, min_rows: int = 50) -> Dataset:
    data = load_price_data(path)
    if len(data) < min_rows:
        raise ValueError("Not enough rows for training.")
    features = build_features(data)
    labels = build_labels(data)
    aligned = features.iloc[:-3]
    labels = labels.iloc[:-3]
    return Dataset(features=aligned, labels=labels)


def iter_recent_rows(data: pd.DataFrame, lookback: int) -> Iterable[pd.Series]:
    for _, row in data.tail(lookback).iterrows():
        yield row
