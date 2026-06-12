from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from preprocessing.config import FAKE_TYPES
from preprocessing.utils import clean_name, short_hash, normalize_dataframe_columns


def detect_manipulation_type(path_text: str) -> str:
    text = str(path_text).lower()

    if "original" in text:
        return "original"

    for fake_type in FAKE_TYPES:
        if fake_type.lower() in text:
            return fake_type

    return "unknown"


def resolve_video_path(raw_root: Path, file_path: str, manipulation_type: str) -> Path:
    file_path = str(file_path).replace("\\", "/").strip()
    path = Path(file_path)

    if path.is_absolute():
        return path

    direct_path = raw_root / path
    if direct_path.exists():
        return direct_path

    fallback_path = raw_root / manipulation_type / path.name
    return fallback_path


def create_video_id(manipulation_type: str, video_path: Path) -> str:
    return clean_name(
        f"{manipulation_type}_{video_path.stem}_{short_hash(str(video_path))}"
    )


def load_ffpp_metadata(raw_root: Path, csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = normalize_dataframe_columns(df)

    if "file_path" not in df.columns:
        raise ValueError(
            "CSV must contain a file path column. Expected column like: File Path"
        )

    rows = []

    for _, row in df.iterrows():
        raw_file_path = row["file_path"]
        manipulation_type = detect_manipulation_type(raw_file_path)

        video_path = resolve_video_path(
            raw_root=raw_root,
            file_path=raw_file_path,
            manipulation_type=manipulation_type,
        )

        if "label" in df.columns:
            label_text = str(row["label"]).upper()
            label = "REAL" if "REAL" in label_text else "FAKE"
        else:
            label = "REAL" if manipulation_type == "original" else "FAKE"

        label_id = 0 if label == "REAL" else 1

        rows.append({
            "video_id": create_video_id(manipulation_type, video_path),
            "video_path": str(video_path),
            "label": label,
            "label_id": label_id,
            "manipulation_type": manipulation_type,
            "frame_count_csv": row.get("frame_count", None),
            "width_csv": row.get("width", None),
            "height_csv": row.get("height", None),
            "codec": row.get("codec", None),
            "file_size_mb": row.get("file_size_mb", None),
        })

    metadata = pd.DataFrame(rows)
    metadata = metadata.drop_duplicates(subset=["video_path"]).reset_index(drop=True)

    return metadata


def is_valid_video(video_path: str) -> bool:
    path = Path(video_path)

    if not path.exists():
        return False

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        cap.release()
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return frame_count > 0


def filter_valid_videos(metadata: pd.DataFrame) -> pd.DataFrame:
    valid_rows = []

    for _, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Checking videos"
    ):
        if is_valid_video(row["video_path"]):
            valid_rows.append(row.to_dict())

    return pd.DataFrame(valid_rows)


def balance_videos(metadata: pd.DataFrame, real_count: int, fake_per_type: int, random_state: int) -> pd.DataFrame:
    real_df = metadata[metadata["label_id"] == 0]

    if len(real_df) == 0:
        raise ValueError("No REAL videos found.")

    real_df = real_df.sample(
        n=min(real_count, len(real_df)),
        random_state=random_state,
    )

    fake_df = metadata[metadata["label_id"] == 1]
    fake_parts = []

    for fake_type in FAKE_TYPES:
        part = fake_df[fake_df["manipulation_type"] == fake_type]

        if len(part) == 0:
            print(f"Warning: no videos found for fake type: {fake_type}")
            continue

        part = part.sample(
            n=min(fake_per_type, len(part)),
            random_state=random_state,
        )

        fake_parts.append(part)

    if len(fake_parts) == 0:
        raise ValueError("No FAKE videos found.")

    balanced = pd.concat([real_df] + fake_parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return balanced