# data/dataset.py

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FaceDatasetFrame(Dataset):
    """
    Dataset for preprocessed face images.

    Expected CSV columns:
        face_path, label_id

    Optional CSV columns:
        video_id, frame_num

    Label convention:
        0 -> REAL
        1 -> FAKE
    """

    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.csv_path = Path(csv_path)
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        required_columns = {"face_path", "label_id"}
        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")

        if len(self.df) == 0:
            raise ValueError(f"No samples found in CSV: {self.csv_path}")

        self.df["label_id"] = self.df["label_id"].astype(int)

        invalid_labels = set(self.df["label_id"].unique()) - {0, 1}
        if invalid_labels:
            raise ValueError(f"Invalid label_id values found: {invalid_labels}")

    def _resolve_image_path(self, face_path: str) -> Path:
        face_path = str(face_path).replace("\\", "/")
        image_path = Path(face_path)

        if image_path.is_absolute():
            return image_path

        return self.root_dir / image_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = self._resolve_image_path(row["face_path"])

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(float(row["label_id"]), dtype=torch.float32)

        sample = {
            "image": image,
            "label": label,
            "face_path": str(row["face_path"]),
        }

        if "video_id" in self.df.columns:
            sample["video_id"] = str(row["video_id"])

        if "frame_num" in self.df.columns:
            sample["frame_num"] = int(row["frame_num"])

        return sample