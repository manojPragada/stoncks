from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    model_path: Path
    data_path: Path
    commission_per_contract: float = 1.0
    refresh_seconds: float = 1.0
    lookback_rows: int = 500
