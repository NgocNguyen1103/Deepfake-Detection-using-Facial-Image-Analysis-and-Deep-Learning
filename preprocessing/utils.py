# preprocessing/utils.py

import hashlib
import re
from pathlib import Path


def clean_name(text: str) -> str:
    text = str(text).replace("\\", "/")
    text = text.replace("/", "_")
    text = re.sub(r"[^a-zA-Z0-9_.-]", "_", text)
    return text


def short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def normalize_column_name(column_name: str) -> str:
    name = str(column_name).strip().lower()
    name = name.replace("(mb)", "mb")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def normalize_dataframe_columns(df):
    new_columns = {}

    for col in df.columns:
        new_columns[col] = normalize_column_name(col)

    return df.rename(columns=new_columns)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)