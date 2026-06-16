# preprocessing/video_processor.py

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.output_writer import append_failed_row, append_frame_row
from preprocessing.utils import clean_name, ensure_dir


def get_uniform_frame_indices(total_frames: int, frames_per_video: int):
    if total_frames <= 0:
        return []

    count = min(total_frames, frames_per_video)
    indices = np.linspace(0, total_frames - 1, count, dtype=int)
    return sorted(set(indices.tolist()))


def process_single_video(
    video_row,
    output_root: Path,
    frames_per_video: int,
    face_detector,
):
    video_path = video_row["video_path"]
    video_id = video_row["video_id"]
    label = video_row["label"]
    label_id = int(video_row["label_id"])
    manipulation_type = video_row["manipulation_type"]
    split = video_row["split"]

    label_folder = "real" if label_id == 0 else "fake"
    save_dir = output_root / split / label_folder
    ensure_dir(save_dir)

    success_count = 0
    failed_count = 0

    def record_failed(frame_number, reason):
        nonlocal failed_count

        failed_row = {
            "video_id": video_id,
            "video_path": video_path,
            "frame_number": frame_number,
            "reason": reason,
        }

        append_failed_row(output_root, failed_row)
        failed_count += 1

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        record_failed(frame_number=None, reason="Cannot open video")
        return success_count, failed_count

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = get_uniform_frame_indices(total_frames, frames_per_video)

        for frame_number in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            success, frame_bgr = cap.read()

            if not success or frame_bgr is None:
                record_failed(frame_number=frame_number, reason="Cannot read frame")
                continue

            bbox = face_detector.detect_best_face(frame_bgr)

            if bbox is None:
                record_failed(frame_number=frame_number, reason="No face detected")
                continue

            face_rgb = face_detector.crop_face(frame_bgr, bbox)

            if face_rgb is None:
                record_failed(frame_number=frame_number, reason="Invalid face crop")
                continue

            sample_id = clean_name(f"{video_id}_frame_{frame_number:06d}")
            image_name = f"{sample_id}.jpg"
            output_path = save_dir / image_name

            face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
            image_written = cv2.imwrite(str(output_path), face_bgr)

            if not image_written:
                record_failed(frame_number=frame_number, reason="Cannot write image")
                continue

            relative_face_path = output_path.relative_to(output_root).as_posix()

            frame_row = {
                "sample_id": sample_id,
                "video_id": video_id,
                "frame_number": frame_number,
                "face_path": relative_face_path,
                "source_video_path": video_path,
                "label": label,
                "label_id": label_id,
                "manipulation_type": manipulation_type,
                "split": split,
                "confidence": bbox["confidence"],
                "bbox_x": bbox["x1"],
                "bbox_y": bbox["y1"],
                "bbox_width": bbox["x2"] - bbox["x1"],
                "bbox_height": bbox["y2"] - bbox["y1"],
                "face_width": face_detector.output_size,
                "face_height": face_detector.output_size,
            }

            append_frame_row(output_root, frame_row)
            success_count += 1

    finally:
        cap.release()

    return success_count, failed_count


def process_dataset(
    video_metadata_split: pd.DataFrame,
    output_root: Path,
    frames_per_video: int,
    face_detector,
):
    total_success = 0
    total_failed = 0

    for _, row in tqdm(
        video_metadata_split.iterrows(),
        total=len(video_metadata_split),
        desc="Processing videos",
    ):
        success_count, failed_count = process_single_video(
            video_row=row,
            output_root=output_root,
            frames_per_video=frames_per_video,
            face_detector=face_detector,
        )

        total_success += success_count
        total_failed += failed_count

    return {
        "success_frames": total_success,
        "failed_frames": total_failed,
    }