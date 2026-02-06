import pandas as pd
from pathlib import Path


def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Load raw CSV data from the data/raw directory.
    """
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / filename

    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    df = pd.read_csv(data_path)
    return dfcrop