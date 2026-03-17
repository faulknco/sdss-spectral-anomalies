"""Load and save human review labels for anomaly candidates."""
from pathlib import Path

import pandas as pd

LABEL_CODES: list[str] = [
    "artifact",
    "low_sn",
    "plausible_oddity",
    "known_rare",
    "unclear",
]

LABEL_DISPLAY_NAMES: dict[str, str] = {
    "artifact": "Instrument/Reduction Artifact",
    "low_sn": "Low S/N Nuisance",
    "plausible_oddity": "Astrophysically Plausible Oddity",
    "known_rare": "Known Rare Subtype",
    "unclear": "Unclear",
}

_SCHEMA_COLUMNS = ["filename", "label", "notes", "timestamp"]


def load_labels(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=_SCHEMA_COLUMNS)


def save_labels(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
