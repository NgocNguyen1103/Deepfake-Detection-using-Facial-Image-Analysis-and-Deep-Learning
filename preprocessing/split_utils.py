import pandas as pd
from sklearn.model_selection import train_test_split


def split_videos(metadata: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, random_state: int) -> pd.DataFrame:
    total_ratio = train_ratio + val_ratio + test_ratio

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    train_df, temp_df = train_test_split(
        metadata,
        test_size=val_ratio + test_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=metadata["label_id"],
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=temp_df["label_id"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    result = pd.concat([train_df, val_df, test_df], ignore_index=True)

    check_video_leakage(result)

    return result


def check_video_leakage(metadata: pd.DataFrame) -> None:
    split_count = metadata.groupby("video_id")["split"].nunique()
    leaked_videos = split_count[split_count > 1]

    if len(leaked_videos) > 0:
        raise ValueError(
            "Data leakage detected. Some videos appear in multiple splits: "
            f"{list(leaked_videos.index[:10])}"
        )