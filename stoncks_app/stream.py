from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import pandas as pd

from stoncks_app.data import load_price_data


def replay_stream(data_path: Path, refresh_seconds: float = 1.0) -> Iterator[pd.Series]:
    data = load_price_data(data_path)
    for _, row in data.iterrows():
        yield row
        time.sleep(refresh_seconds)
