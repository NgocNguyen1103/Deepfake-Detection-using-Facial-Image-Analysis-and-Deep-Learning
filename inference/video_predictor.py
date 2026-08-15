# inference/video_predictor.py
# Complete pipeline for video-level deepfake detection

import sys
from pathlib import Path
import cv2
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.efficientnet_baseline import EfficientNetB0Baseline
from preprocessing.face_detector import FaceDetector
from torchvision import transforms


class VideoPredictor:
    """
    Complete pipeline for video-level deepfake detection.

    Pipeline steps:
    1. Frame extraction (every 10 seconds get 1 frame, if video < 10 seconds get all frames)
    2. Face cropping using RetinaFace
    3. Normalization
    4. Model inference with trained EfficientNet-B0
    5. Frame-level fake_score calculation
    6. Aggregation by mean
    7. Final prediction using threshold
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        threshold: float = 0.5,
        min_confidence: float = 0.90,
        margin_ratio: float = 0.20,
        frame_interval: int = 10,  # seconds
    ):
        """
        Initialize the video predictor.

        Args:
            model_path: Path to trained model weights (enhanced-best.pth)
            device: Device to run inference on ('cuda' or 'cpu')
            threshold: Threshold for final binary prediction
            min_confidence: Minimum confidence for face detection
            margin_ratio: Margin ratio for face cropping
            frame_interval: Interval in seconds for frame extraction
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.frame_interval = frame_interval

        # Initialize face detector
        self.face_detector = FaceDetector(
            min_confidence=min_confidence,
            margin_ratio=margin_ratio,
            output_size=224,
        )

        # Initialize normalization transform (same as training)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Load model
        if model_path is None:
            model_path = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_enhanced_best.pth"

        self.model = self._load_model(model_path)
        self.model.eval()

        print(f"Video Predictor initialized on device: {self.device}")
        print(f"Model loaded from: {model_path}")
        print(f"Frame extraction interval: {frame_interval} seconds")
        print(f"Prediction threshold: {threshold}")

    def _load_model(self, model_path: str) -> EfficientNetB0Baseline:
        """Load trained model with weights."""
        model = EfficientNetB0Baseline(pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        return model

    def extract_frames(self, video_path: str) -> List[Tuple[int, np.ndarray]]:
        """
        Extract frames from video at specified interval.

        Args:
            video_path: Path to video file

        Returns:
            List of (frame_number, frame) tuples
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"Video properties - FPS: {fps:.2f}, Total frames: {total_frames}, Duration: {duration:.2f}s")

        frames = []
        frame_interval_frames = int(fps * self.frame_interval) if fps > 0 else 1

        # Calculate which frames to extract
        if duration < self.frame_interval:
            # If video is less than interval, extract all frames
            frame_numbers = list(range(total_frames))
            print(f"Video duration ({duration:.2f}s) < interval ({self.frame_interval}s), extracting all {len(frame_numbers)} frames")
        else:
            # Extract frames at specified interval
            frame_numbers = list(range(0, total_frames, frame_interval_frames))
            print(f"Extracting {len(frame_numbers)} frames at {self.frame_interval}s interval")

        # Extract frames
        for frame_num in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if ret and frame is not None:
                frames.append((frame_num, frame))

        cap.release()

        print(f"Successfully extracted {len(frames)} frames")
        return frames

    def detect_and_crop_faces(self, frames: List[Tuple[int, np.ndarray]]) -> List[Dict]:
        """
        Detect and crop faces from extracted frames.

        Args:
            frames: List of (frame_number, frame) tuples

        Returns:
            List of dictionaries containing frame info and cropped faces
        """
        face_data = []

        print("Detecting and cropping faces from frames...")
        for frame_num, frame in tqdm(frames):
            # Detect best face
            bbox = self.face_detector.detect_best_face(frame)

            if bbox is not None:
                # Crop face
                face_rgb = self.face_detector.crop_face(frame, bbox)

                if face_rgb is not None:
                    face_data.append({
                        'frame_number': frame_num,
                        'face_rgb': face_rgb,
                        'bbox': bbox,
                        'face_detected': True
                    })
                else:
                    face_data.append({
                        'frame_number': frame_num,
                        'face_rgb': None,
                        'bbox': bbox,
                        'face_detected': False
                    })
            else:
                face_data.append({
                    'frame_number': frame_num,
                    'face_rgb': None,
                    'bbox': None,
                    'face_detected': False
                })

        detected_count = sum(1 for f in face_data if f['face_detected'])
        print(f"Successfully detected and cropped faces in {detected_count}/{len(frames)} frames")

        return face_data

    def preprocess_faces(self, face_data: List[Dict]) -> List[Dict]:
        """
        Preprocess face images for model input.

        Args:
            face_data: List of dictionaries containing face information

        Returns:
            Updated face_data with preprocessed tensors
        """
        print("Preprocessing face images...")

        for data in tqdm(face_data):
            if data['face_detected'] and data['face_rgb'] is not None:
                # Apply normalization transform
                face_tensor = self.transform(data['face_rgb'])
                data['face_tensor'] = face_tensor
            else:
                data['face_tensor'] = None

        return face_data

    def calculate_frame_scores(self, face_data: List[Dict]) -> List[Dict]:
        """
        Calculate fake scores for each frame using the model.

        Args:
            face_data: List of dictionaries containing preprocessed face data

        Returns:
            Updated face_data with fake scores
        """
        print("Calculating frame-level fake scores...")

        with torch.no_grad():
            for data in tqdm(face_data):
                if data['face_tensor'] is not None:
                    # Prepare input
                    face_input = data['face_tensor'].unsqueeze(0).to(self.device)

                    # Model inference
                    logit = self.model(face_input)

                    # Calculate fake score using sigmoid
                    fake_score = torch.sigmoid(logit).item()

                    data['fake_score'] = fake_score
                    data['prediction'] = "FAKE" if fake_score >= self.threshold else "REAL"
                else:
                    # No face detected - assign neutral score
                    data['fake_score'] = 0.5
                    data['prediction'] = "UNCERTAIN"

        return face_data

    def aggregate_scores(self, face_data: List[Dict]) -> Dict:
        """
        Aggregate frame-level scores to video-level prediction.

        Args:
            face_data: List of dictionaries containing frame scores

        Returns:
            Dictionary containing aggregated results
        """
        # Filter only frames with valid detections
        valid_frames = [f for f in face_data if f['face_detected']]

        if not valid_frames:
            return {
                'aggregate_score': 0.5,
                'final_prediction': 'UNCERTAIN',
                'total_frames': len(face_data),
                'detected_frames': 0,
                'detection_rate': 0.0,
                'message': 'No faces detected in any frames'
            }

        # Extract fake scores
        fake_scores = [f['fake_score'] for f in valid_frames]

        # Calculate aggregate score (mean)
        aggregate_score = np.mean(fake_scores)

        # Make final prediction
        final_prediction = "FAKE" if aggregate_score >= self.threshold else "REAL"

        # Calculate statistics
        std_score = np.std(fake_scores)
        min_score = np.min(fake_scores)
        max_score = np.max(fake_scores)

        result = {
            'aggregate_score': aggregate_score,
            'final_prediction': final_prediction,
            'total_frames': len(face_data),
            'detected_frames': len(valid_frames),
            'detection_rate': len(valid_frames) / len(face_data),
            'score_statistics': {
                'mean': aggregate_score,
                'std': std_score,
                'min': min_score,
                'max': max_score,
            },
            'individual_scores': fake_scores,
        }

        return result

    def predict_video(self, video_path: str, return_details: bool = False) -> Dict:
        """
        Complete pipeline for video-level prediction.

        Args:
            video_path: Path to video file
            return_details: Whether to return detailed frame-by-frame results

        Returns:
            Dictionary containing prediction results
        """
        print(f"\n{'='*60}")
        print(f"Processing video: {video_path}")
        print(f"{'='*60}\n")

        try:
            # Step 1: Extract frames
            print("Step 1: Frame Extraction")
            frames = self.extract_frames(video_path)

            if not frames:
                return {
                    'error': 'No frames extracted from video',
                    'final_prediction': 'ERROR'
                }

            # Step 2: Detect and crop faces
            print("\nStep 2: Face Detection and Cropping")
            face_data = self.detect_and_crop_faces(frames)

            # Step 3: Preprocess faces
            print("\nStep 3: Preprocessing")
            face_data = self.preprocess_faces(face_data)

            # Step 4: Calculate frame scores
            print("\nStep 4: Frame-level Scoring")
            face_data = self.calculate_frame_scores(face_data)

            # Step 5: Aggregate scores
            print("\nStep 5: Score Aggregation")
            results = self.aggregate_scores(face_data)

            # Add additional information
            results['video_path'] = video_path
            results['threshold'] = self.threshold

            if return_details:
                results['frame_details'] = face_data

            # Print summary
            self._print_summary(results)

            return results

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return {
                'error': str(e),
                'final_prediction': 'ERROR'
            }

    def _print_summary(self, results: Dict):
        """Print prediction summary."""
        print(f"\n{'='*60}")
        print("PREDICTION SUMMARY")
        print(f"{'='*60}")

        if 'error' in results:
            print(f"Error: {results['error']}")
            return

        print(f"Video: {results.get('video_path', 'Unknown')}")
        print(f"Total frames processed: {results['total_frames']}")
        print(f"Frames with detected faces: {results['detected_frames']}")
        print(f"Face detection rate: {results['detection_rate']:.2%}")

        if 'score_statistics' in results:
            stats = results['score_statistics']
            print(f"\nFrame-level score statistics:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")

        print(f"\nAggregate fake score: {results['aggregate_score']:.4f}")
        print(f"Final prediction: {results['final_prediction']}")
        print(f"Threshold: {results['threshold']:.4f}")
        print(f"{'='*60}\n")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Video-level deepfake detection pipeline"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model weights (default: checkpoints/efficientnet_b0_enhanced_best.pth)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary prediction (default: 0.5)"
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=10,
        help="Frame extraction interval in seconds (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Return detailed frame-by-frame results"
    )

    args = parser.parse_args()

    # Initialize predictor
    predictor = VideoPredictor(
        model_path=args.model,
        device=args.device,
        threshold=args.threshold,
        frame_interval=args.frame_interval,
    )

    # Process video
    results = predictor.predict_video(
        args.video_path,
        return_details=args.verbose
    )

    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()