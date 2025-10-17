import pandas as pd
from pathlib import Path

# Resolve CSV paths relative to this file: codebase/codebase/ → parent is codebase/
_MODULE_DIR = Path(__file__).resolve().parent
_CSV_DIR = _MODULE_DIR.parent
_MEN_CSV = _CSV_DIR / "resultmen.csv"
_WOMEN_CSV = _CSV_DIR / "resultwomen.csv"

_men_df = None
_women_df = None


def _get_df(gender: str) -> pd.DataFrame:
    global _men_df, _women_df
    if gender.lower().startswith("m"):
        if _men_df is None:
            _men_df = pd.read_csv(_MEN_CSV)
        return _men_df
    else:
        if _women_df is None:
            _women_df = pd.read_csv(_WOMEN_CSV)
        return _women_df


def fetch_suggestions(gender: str, face_shape: str, interocular_category: str, nose_category: str) -> dict:
    df = _get_df(gender)
    row = df[(df["face_shape"].str.lower() == face_shape.lower()) &
             (df["interocular_category"].str.lower() == interocular_category.lower()) &
             (df["nose_category"].str.lower() == nose_category.lower())]
    if row.empty:
        return {"hairstyle_suggestions": None, "beard_suggestions": None}
    row = row.iloc[0]
    result = {"hairstyle_suggestions": row.get("hairstyle_suggestions")}
    if gender.lower().startswith("m") and "beard_suggestions" in row:
        result["beard_suggestions"] = row.get("beard_suggestions")
    else:
        result["beard_suggestions"] = None
    return result





