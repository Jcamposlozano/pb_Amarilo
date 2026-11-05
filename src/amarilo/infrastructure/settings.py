import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(model_path=Path(os.environ.get("MODEL_PATH", "models/model.joblib")))
