from pathlib import Path
import csv
import time
from collections import defaultdict

import pandas as pd


FRAME_METADATA_COLUMNS = [
    "sample_id",
    "video_id",
    "frame_number",
    "face_path",
    "source_video_path",
    "label",
    "label_id",
    "manipulation_type",
    "split",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "face_width",
    "face_height",
]

FAILED_FRAME_COLUMNS = [
    "video_id",
    "video_path",
    "frame_number",
    "reason",
]

SPLITS = ["train", "val", "test"]

# Rows that could not be written because the CSV file was locked.
_PENDING_ROWS = defaultdict(list)

# Store CSV columns by file path so we can flush later.
_PENDING_COLUMNS = {}

# Avoid printing the same warning too many times.
_WARNED_LOCKED_FILES = set()


def create_empty_csv(csv_path: Path, columns: list[str]) -> None:
    """
    Create a CSV file with only the header row.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()


def initialize_output_files(output_root: Path, video_metadata_split: pd.DataFrame) -> None:
    """
    Create output folders and CSV files before processing frames.

    frame_metadata.csv, failed_frames.csv, train.csv, val.csv, test.csv
    are created first with header only.

    video_metadata_split.csv is written immediately after video split.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        (output_root / split / "real").mkdir(parents=True, exist_ok=True)
        (output_root / split / "fake").mkdir(parents=True, exist_ok=True)

    video_metadata_split.to_csv(
        output_root / "video_metadata_split.csv",
        index=False,
    )

    create_empty_csv(output_root / "frame_metadata.csv", FRAME_METADATA_COLUMNS)
    create_empty_csv(output_root / "failed_frames.csv", FAILED_FRAME_COLUMNS)

    for split in SPLITS:
        create_empty_csv(output_root / f"{split}.csv", FRAME_METADATA_COLUMNS)


def normalize_row(row: dict, columns: list[str]) -> dict:
    """
    Keep only known columns and preserve column order.
    """
    return {column: row.get(column, None) for column in columns}


def write_rows_to_csv(csv_path: Path, rows: list[dict], columns: list[str]) -> None:
    """
    Write rows to CSV.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writerows(rows)
        file.flush()


def flush_pending_rows(csv_path: Path) -> None:
    """
    Try to write pending rows for one CSV file.
    If the file is still locked, keep the rows in memory and continue
    """
    key = str(csv_path)

    if key not in _PENDING_ROWS:
        return

    if len(_PENDING_ROWS[key]) == 0:
        return

    columns = _PENDING_COLUMNS.get(key)

    if columns is None:
        return

    pending_rows = _PENDING_ROWS[key]

    try:
        write_rows_to_csv(csv_path, pending_rows, columns)
        _PENDING_ROWS[key] = []

        if key in _WARNED_LOCKED_FILES:
            print(f"[CSV unlocked] Pending rows flushed to: {csv_path}")
            _WARNED_LOCKED_FILES.remove(key)

    except PermissionError:
        # Still locked. Keep pending rows.
        if key not in _WARNED_LOCKED_FILES:
            print(
                f"[CSV locked] Cannot write to {csv_path}. "
                f"Close the file if it is opened in Excel/WPS. "
                f"Rows will be written later."
            )
            _WARNED_LOCKED_FILES.add(key)


def append_row_to_csv(csv_path: Path, row: dict, columns: list[str]) -> None:
    """
    Append one row to CSV.

    If the CSV file is locked by Excel/WPS, the row is stored in memory
    and will be written later when the file becomes available again.
    """
    key = str(csv_path)
    normalized_row = normalize_row(row, columns)

    _PENDING_COLUMNS[key] = columns

    # First try to flush old pending rows.
    flush_pending_rows(csv_path)

    try:
        write_rows_to_csv(csv_path, [normalized_row], columns)

    except PermissionError:
        _PENDING_ROWS[key].append(normalized_row)

        if key not in _WARNED_LOCKED_FILES:
            print(
                f"[CSV locked] Cannot write to {csv_path}. "
                f"Close the file if it is opened in Excel/WPS. "
                f"Rows will be written later."
            )
            _WARNED_LOCKED_FILES.add(key)


def append_frame_row(output_root: Path, frame_row: dict) -> None:
    """
    Append one successful frame to:
    - frame_metadata.csv
    - train.csv / val.csv / test.csv
    """
    append_row_to_csv(
        output_root / "frame_metadata.csv",
        frame_row,
        FRAME_METADATA_COLUMNS,
    )

    split = frame_row.get("split")

    if split in SPLITS:
        append_row_to_csv(
            output_root / f"{split}.csv",
            frame_row,
            FRAME_METADATA_COLUMNS,
        )


def append_failed_row(output_root: Path, failed_row: dict) -> None:
    """
    Append one failed frame/video record to failed_frames.csv.
    """
    append_row_to_csv(
        output_root / "failed_frames.csv",
        failed_row,
        FAILED_FRAME_COLUMNS,
    )


def flush_all_pending_rows(max_retries: int = 5, sleep_seconds: float = 1.0) -> None:
    """
    Try to flush all pending rows before the program exits.

    If the user keeps CSV files opened in Excel/WPS until the end,
    some rows may still remain pending.
    """
    for attempt in range(max_retries + 1):
        total_pending = get_total_pending_rows()

        if total_pending == 0:
            return

        for path_text in list(_PENDING_ROWS.keys()):
            csv_path = Path(path_text)
            flush_pending_rows(csv_path)

        if get_total_pending_rows() == 0:
            return

        if attempt < max_retries:
            time.sleep(sleep_seconds)

    total_pending = get_total_pending_rows()

    if total_pending > 0:
        print(
            f"[Warning] {total_pending} CSV rows are still pending because "
            f"some CSV files are locked. Close Excel/WPS and run again if needed."
        )


def get_total_pending_rows() -> int:
    return sum(len(rows) for rows in _PENDING_ROWS.values())


def load_frame_metadata(output_root: Path) -> pd.DataFrame:
    frame_metadata_path = output_root / "frame_metadata.csv"

    if not frame_metadata_path.exists():
        return pd.DataFrame(columns=FRAME_METADATA_COLUMNS)

    return pd.read_csv(frame_metadata_path)


def print_processing_summary_from_csv(output_root: Path) -> None:
    frame_metadata = load_frame_metadata(output_root)
    print_processing_summary(frame_metadata)


def save_outputs(
    output_root: Path,
    video_metadata_split: pd.DataFrame,
    frame_metadata: pd.DataFrame,
    failed_frames: pd.DataFrame,
) -> None:
    
    output_root.mkdir(parents=True, exist_ok=True)

    video_metadata_split.to_csv(
        output_root / "video_metadata_split.csv",
        index=False,
    )

    frame_metadata.to_csv(
        output_root / "frame_metadata.csv",
        index=False,
    )

    failed_frames.to_csv(
        output_root / "failed_frames.csv",
        index=False,
    )

    for split in SPLITS:
        split_df = frame_metadata[frame_metadata["split"] == split].copy()
        split_df.to_csv(output_root / f"{split}.csv", index=False)


def print_processing_summary(frame_metadata: pd.DataFrame) -> None:
    print("\n========== Processing Summary ==========")

    if len(frame_metadata) == 0:
        print("No face images were created.")
        return

    summary = frame_metadata.groupby(
        ["split", "label", "manipulation_type"]
    ).size()

    print(summary)