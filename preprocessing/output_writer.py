from pathlib import Path
import csv
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

def create_empty_csv(csv_path: Path, columns: list[str]) -> None:
    """
    Create empty CSV file with input header row
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()


def initialize_output_files(output_root: Path, video_metadata_split: pd.DataFrame) -> None:
    """
    Create empty requiired output folders and CSV files before processing frames.
    frame_metadata.csv, failed_frames.csv, train.csv, val.csv, test.csv
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
        

def append_row_to_csv(csv_path: Path, row: dict, columns: list[str]) -> None:
    """
    Append one row to CSV and close the file immediately.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_row = {column: row.get(column, None) for column in columns}

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writerow(normalized_row)
        file.flush()


def append_frame_row(output_root: Path, frame_row: dict) -> None:
    """
    Append one successful frame to:
    - frame_metadata.csv
    - train.csv / val.csv / test.csv
    """
    append_row_to_csv(output_root / "frame_metadata.csv", frame_row, FRAME_METADATA_COLUMNS)

    split = frame_row.get("split")

    if split in SPLITS:
        append_row_to_csv(output_root / f"{split}.csv", frame_row, FRAME_METADATA_COLUMNS)


def append_failed_row(output_root: Path, failed_row: dict) -> None:
    """
    Append one failed frame/video record to failed_frames.csv.
    """
    append_row_to_csv(output_root / "failed_frames.csv", failed_row, FAILED_FRAME_COLUMNS)


def load_frame_metadata(output_root: Path) -> pd.DataFrame:
    frame_metadata_path = output_root / "frame_metadata.csv"

    if not frame_metadata_path.exists():
        return pd.DataFrame(columns=FRAME_METADATA_COLUMNS)

    return pd.read_csv(frame_metadata_path)


def print_processing_summary_from_csv(output_root: Path) -> None:
    frame_metadata = load_frame_metadata(output_root)
    print_processing_summary(frame_metadata)



def save_outputs(output_root: Path, video_metadata_split: pd.DataFrame, frame_metadata: pd.DataFrame, failed_frames: pd.DataFrame) -> None:
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