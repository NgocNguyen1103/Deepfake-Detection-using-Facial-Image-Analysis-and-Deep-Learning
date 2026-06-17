# run_preprocess.py

import argparse
from pathlib import Path

from preprocessing.config import (
    DEFAULT_FRAMES_PER_VIDEO,
    DEFAULT_MARGIN_RATIO,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_OUTPUT_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
)
from preprocessing.face_detector import FaceDetector
from preprocessing.metadata_loader import (
    balance_videos,
    filter_valid_videos,
    load_ffpp_metadata,
)
from preprocessing.output_writer import (
    flush_all_pending_rows,
    initialize_output_files,
    load_processed_frame_keys,
    load_video_metadata_split,
    print_processing_summary_from_csv,
)

from preprocessing.split_utils import split_videos
from preprocessing.video_processor import process_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess FaceForensics++ video dataset into cropped face images."
    )

    parser.add_argument(
        "--raw_root",
        required=True,
        help="Root folder of raw FaceForensics++ dataset.",
    )

    parser.add_argument(
        "--csv_path",
        required=True,
        help="Path to FF++ metadata CSV file.",
    )

    parser.add_argument(
        "--output_root",
        required=True,
        help="Output folder for preprocessed dataset.",
    )

    parser.add_argument(
        "--real_count",
        type=int,
        default=300,
        help="Number of REAL videos to use.",
    )

    parser.add_argument(
        "--fake_per_type",
        type=int,
        default=50,
        help="Number of FAKE videos per manipulation type.",
    )

    parser.add_argument(
        "--frames_per_video",
        type=int,
        default=DEFAULT_FRAMES_PER_VIDEO,
        help="Number of frames sampled per video.",
    )

    parser.add_argument(
        "--min_confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Minimum face detection confidence.",
    )

    parser.add_argument(
        "--margin_ratio",
        type=float,
        default=DEFAULT_MARGIN_RATIO,
        help="Margin ratio added around detected face bbox.",
    )

    parser.add_argument(
        "--output_size",
        type=int,
        default=DEFAULT_OUTPUT_SIZE,
        help="Output face image size.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume preprocessing from existing CSV files instead of starting from scratch.",
    )

    parser.add_argument(
        "--retry_failed_frames",
        action="store_true",
        help="In resume mode, retry failed frames instead of skipping them.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    raw_root = Path(args.raw_root)
    csv_path = Path(args.csv_path)
    output_root = Path(args.output_root)

    if args.resume and (output_root / "video_metadata_split.csv").exists():
        print("Resume mode: Load existing video split metadata")
        video_metadata_split = load_video_metadata_split(output_root)

    else:
        print("Step 1: Load CSV metadata")
        metadata = load_ffpp_metadata(
            raw_root=raw_root,
            csv_path=csv_path,
        )

        print("Step 2: Filter valid videos")
        metadata = filter_valid_videos(metadata)

        print("Step 3: Balance REAL and FAKE videos")
        metadata = balance_videos(
            metadata=metadata,
            real_count=args.real_count,
            fake_per_type=args.fake_per_type,
            random_state=args.random_state,
        )

        print("Step 4: Split videos into train/val/test")
        video_metadata_split = split_videos(
            metadata=metadata,
            train_ratio=DEFAULT_TRAIN_RATIO,
            val_ratio=DEFAULT_VAL_RATIO,
            test_ratio=DEFAULT_TEST_RATIO,
            random_state=args.random_state,
        )

    print("Step 5: Create or reuse output CSV files")
    initialize_output_files(
        output_root=output_root,
        video_metadata_split=video_metadata_split,
        resume=args.resume,
    )

    print("Step 6: Initialize RetinaFace detector")
    face_detector = FaceDetector(
        min_confidence=args.min_confidence,
        margin_ratio=args.margin_ratio,
        output_size=args.output_size,
    )

    processed_frame_keys = set()

    if args.resume:
        print("Resume mode: Load processed frame keys")

        processed_frame_keys = load_processed_frame_keys(
            output_root=output_root,
            include_failed_frames=not args.retry_failed_frames,
        )

        print(f"Already processed frames: {len(processed_frame_keys)}")

    print("Step 7: Extract frames, detect faces, crop faces, append CSV rows")

    interrupted = False
    processing_stats = None

    try:
        processing_stats = process_dataset(
            video_metadata_split=video_metadata_split,
            output_root=output_root,
            frames_per_video=args.frames_per_video,
            face_detector=face_detector,
            processed_frame_keys=processed_frame_keys,
        )

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user.")

    finally:
        print("\nFlushing pending CSV rows...")
        flush_all_pending_rows(max_retries=10, sleep_seconds=1.0)

    print_processing_summary_from_csv(output_root)

    if interrupted:
        print("\nStopped before finishing.")
        print(f"Progress was saved to: {output_root}")
        print("Run the same command with --resume to continue.")
    else:
        print("\nDone.")

        if processing_stats is not None:
            print(f"Success frames: {processing_stats['success_frames']}")
            print(f"Failed frames: {processing_stats['failed_frames']}")
            print(f"Skipped frames: {processing_stats['skipped_frames']}")

        print(f"Processed dataset saved to: {output_root}")


if __name__ == "__main__":
    main()