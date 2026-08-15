import pandas as pd
import os
import shutil
from pathlib import Path
import random

# Set random seed for reproducibility
random.seed(42)

# Define paths
BASE_DIR = r"D:\M1\M1 Internship"
CSV_PATH = os.path.join(BASE_DIR, "preprocessed_dataset", "test.csv")
SOURCE_VIDEO_DIR = os.path.join(BASE_DIR, "FaceForensics++_C23")
TARGET_DIR = os.path.join(BASE_DIR, "test-video-level-dataset")

# Create target directory
os.makedirs(TARGET_DIR, exist_ok=True)
print(f"Created target directory: {TARGET_DIR}")

# Read the CSV file
df = pd.read_csv(CSV_PATH)

# Get unique video information (one row per video)
video_info = df.groupby('video_id').agg({
    'label': 'first',
    'manipulation_type': 'first',
    'source_video_path': 'first'
}).reset_index()

print(f"\nTotal unique videos: {len(video_info)}")
print(f"REAL videos: {len(video_info[video_info['label'] == 'REAL'])}")
print(f"FAKE videos: {len(video_info[video_info['label'] == 'FAKE'])}")

# Calculate required samples
TOTAL_VIDEOS = 100
REAL_VIDEOS = 50  # 50%
FAKE_VIDEOS = 50   # 50%
FAKE_TYPES = ['Deepfakes', 'Face2Face', 'FaceSwap', 'FaceShifter', 'NeuralTextures']
VIDEOS_PER_FAKE_TYPE = 10  # 10% each (10 out of 100)

print(f"\nTarget distribution:")
print(f"- REAL videos: {REAL_VIDEOS}")
print(f"- FAKE videos: {FAKE_VIDEOS}")
for fake_type in FAKE_TYPES:
    print(f"  - {fake_type}: {VIDEOS_PER_FAKE_TYPE} videos")

# Sample REAL videos
real_videos = video_info[video_info['label'] == 'REAL']
sampled_real = real_videos.sample(n=min(REAL_VIDEOS, len(real_videos)), random_state=42)

# Sample FAKE videos by manipulation type
sampled_fake = pd.DataFrame()
for fake_type in FAKE_TYPES:
    fake_type_videos = video_info[
        (video_info['label'] == 'FAKE') &
        (video_info['manipulation_type'] == fake_type)
    ]

    available = len(fake_type_videos)
    to_sample = min(VIDEOS_PER_FAKE_TYPE, available)

    if to_sample > 0:
        sampled = fake_type_videos.sample(n=to_sample, random_state=42)
        sampled_fake = pd.concat([sampled_fake, sampled])
        print(f"Sampled {to_sample}/{available} videos for {fake_type}")
    else:
        print(f"Warning: No videos available for {fake_type}")

# Combine sampled videos
sampled_videos = pd.concat([sampled_real, sampled_fake])

print(f"\nTotal sampled videos: {len(sampled_videos)}")
print(f"REAL: {len(sampled_videos[sampled_videos['label'] == 'REAL'])}")
print(f"FAKE: {len(sampled_videos[sampled_videos['label'] == 'FAKE'])}")

# Create subdirectories in target folder
os.makedirs(os.path.join(TARGET_DIR, "real"), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, "fake"), exist_ok=True)

# Copy videos
copied_count = 0
failed_count = 0

for _, row in sampled_videos.iterrows():
    source_path = row['source_video_path']
    video_id = row['video_id']
    label = row['label']

    # Check if source file exists
    if not os.path.exists(source_path):
        print(f"Warning: Source file not found: {source_path}")
        failed_count += 1
        continue

    # Determine target subdirectory
    target_subdir = "real" if label == "REAL" else "fake"
    target_path = os.path.join(TARGET_DIR, target_subdir, f"{video_id}.mp4")

    # Copy video
    try:
        shutil.copy2(source_path, target_path)
        copied_count += 1
        print(f"Copied: {video_id} -> {target_subdir}/")
    except Exception as e:
        print(f"Error copying {video_id}: {e}")
        failed_count += 1

print(f"\n=== Summary ===")
print(f"Successfully copied: {copied_count}/{len(sampled_videos)} videos")
print(f"Failed: {failed_count}")
print(f"\nDataset created at: {TARGET_DIR}")
print(f"Real videos: {len(os.listdir(os.path.join(TARGET_DIR, 'real')))}")
print(f"Fake videos: {len(os.listdir(os.path.join(TARGET_DIR, 'fake')))}")

# Save the metadata CSV
metadata_path = os.path.join(TARGET_DIR, "metadata.csv")
sampled_videos.to_csv(metadata_path, index=False)
print(f"\nMetadata saved to: {metadata_path}")