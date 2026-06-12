from pathlib import Path

import pandas as pd


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

    for split in ["train", "val", "test"]:
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