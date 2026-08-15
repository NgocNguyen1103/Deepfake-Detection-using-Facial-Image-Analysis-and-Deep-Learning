# scripts/evaluate_cbam_model_video_level.py
# Video-level evaluation script for EfficientNet-CBAM model

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from inference.video_predictor import VideoPredictor
from models.efficientnet_cbam import create_efficientnet_cbam
from models.cbam_modules import CBAMConfig

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ====================================================================
# CBAM MODEL CONFIGURATION
# ====================================================================

# Path to CBAM model checkpoint
CBAM_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "experiments" / "cbam_20260630_235749" / "efficientnet_cbam_best.pth"

# CBAM configuration
CBAM_CONFIG_PATH = PROJECT_ROOT / "configs" / "cbam_config.json"

# ====================================================================
# EVALUATION SETTINGS
# ====================================================================

DATASET_PATH = "D:/M1/M1 Internship/test-video-level-dataset"
OUTPUT_BASE_DIR = PROJECT_ROOT / "results" / "evaluation" / "efficientnet_cbam_acc_loss"
DEVICE = "cuda"  # Options: "cuda" or "cpu"
THRESHOLD = 0.5

# ====================================================================


class CBAMVideoPredictor(VideoPredictor):
    """
    Extended VideoPredictor for EfficientNet-CBAM model.

    Overrides the model loading to use CBAM architecture instead of baseline.
    """

    def __init__(self, model_path: str, config_path: str = None, device: str = "cuda",
                 threshold: float = 0.5, min_confidence: float = 0.90,
                 margin_ratio: float = 0.20, frame_interval: int = 10):
        """
        Initialize CBAM video predictor.

        Args:
            model_path: Path to trained CBAM model weights
            config_path: Path to CBAM configuration JSON file
            device: Device to run inference on ('cuda' or 'cpu')
            threshold: Threshold for final binary prediction
            min_confidence: Minimum confidence for face detection
            margin_ratio: Margin ratio for face cropping
            frame_interval: Interval in seconds for frame extraction
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.frame_interval = frame_interval
        self.model_path = model_path
        self.config_path = config_path

        # Initialize face detector (inherited from VideoPredictor)
        from preprocessing.face_detector import FaceDetector
        self.face_detector = FaceDetector(
            min_confidence=min_confidence,
            margin_ratio=margin_ratio,
            output_size=224,
        )

        # Initialize normalization transform (same as training)
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Load CBAM model
        self.model = self._load_cbam_model(model_path, config_path)
        self.model.eval()

        print(f"CBAM Video Predictor initialized on device: {self.device}")
        print(f"Model loaded from: {model_path}")
        print(f"Frame extraction interval: {frame_interval} seconds")
        print(f"Prediction threshold: {threshold}")

    def _load_cbam_model(self, model_path: str, config_path: str = None):
        """Load trained CBAM model with weights."""
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

            cbam_variant = config.get('model', {}).get('cbam_variant', 'default')
            dropout = config.get('model', {}).get('dropout', 0.5)
            pretrained = config.get('model', {}).get('pretrained', True)

            print(f"Loading CBAM model with config:")
            print(f"  Variant: {cbam_variant}")
            print(f"  Dropout: {dropout}")
            print(f"  Pretrained: {pretrained}")
        else:
            print("Using default CBAM configuration")
            cbam_variant = 'default'
            dropout = 0.5
            pretrained = True

        # Create CBAM model
        model = create_efficientnet_cbam(
            pretrained=pretrained,
            cbam_variant=cbam_variant,
            dropout=dropout
        )

        # Load checkpoint weights
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model from 'model_state_dict' key")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded model from 'state_dict' key")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model directly as state dict")

        model = model.to(self.device)
        return model


class CBAMModelEvaluator:
    """
    Evaluate EfficientNet-CBAM model for video-level deepfake detection.
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        dataset_path: str,
        output_base_dir: str,
        threshold: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize the CBAM model evaluator.

        Args:
            model_path: Path to CBAM model checkpoint
            config_path: Path to CBAM configuration JSON
            dataset_path: Path to test-video-level-dataset
            output_base_dir: Base directory for evaluation results
            threshold: Classification threshold
            device: Device for inference (cuda/cpu)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.dataset_path = Path(dataset_path)
        self.output_base_dir = Path(output_base_dir)
        self.threshold = threshold
        self.device = device

        # Create CBAM-specific output directory
        self.output_dir = self.output_base_dir / "efficientnet_cbam"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"CBAM Model Evaluator initialized")
        print(f"Model: {model_path}")
        print(f"Output directory: {self.output_dir}")

    def load_video_dataset(self):
        """Load video dataset and extract ground truth labels."""
        print("\n" + "="*60)
        print("Loading Video Dataset")
        print("="*60)

        videos = []

        # Load real videos
        real_dir = self.dataset_path / "real"
        if real_dir.exists():
            real_videos = list(real_dir.glob("*.mp4"))
            for video_path in real_videos:
                videos.append({
                    'video_path': str(video_path),
                    'video_id': video_path.stem,
                    'true_label': 'REAL',
                    'label_value': 0
                })
            print(f"Found {len(real_videos)} real videos")

        # Load fake videos
        fake_dir = self.dataset_path / "fake"
        if fake_dir.exists():
            fake_videos = list(fake_dir.glob("*.mp4"))
            for video_path in fake_videos:
                videos.append({
                    'video_path': str(video_path),
                    'video_id': video_path.stem,
                    'true_label': 'FAKE',
                    'label_value': 1
                })
            print(f"Found {len(fake_videos)} fake videos")

        print(f"\nTotal videos to process: {len(videos)}")
        print(f"REAL: {sum(1 for v in videos if v['true_label'] == 'REAL')}")
        print(f"FAKE: {sum(1 for v in videos if v['true_label'] == 'FAKE')}")

        return videos

    def initialize_predictor(self):
        """Initialize CBAM video predictor."""
        print("\n" + "="*60)
        print("Initializing CBAM Model Predictor")
        print("="*60)

        try:
            predictor = CBAMVideoPredictor(
                model_path=self.model_path,
                config_path=self.config_path,
                device=self.device,
                threshold=self.threshold,
                frame_interval=1
            )
            print(f"✓ Loaded CBAM model from: {self.model_path}")
            return predictor

        except Exception as e:
            print(f"✗ Error loading CBAM model: {e}")
            raise

    def process_videos(self, videos, predictor):
        """Process all videos with the CBAM model."""
        print(f"\n{'='*60}")
        print(f"Processing videos with EfficientNet-CBAM")
        print(f"{'='*60}")

        results = []

        for video_info in tqdm(videos, desc="Processing videos"):
            video_path = video_info['video_path']

            try:
                # Get prediction from CBAM predictor
                prediction_result = predictor.predict_video(
                    video_path,
                    return_details=False
                )

                # Extract key information
                if 'error' not in prediction_result:
                    result = {
                        'video_id': video_info['video_id'],
                        'video_path': video_path,
                        'true_label': video_info['true_label'],
                        'true_value': video_info['label_value'],
                        'predicted_label': prediction_result['final_prediction'],
                        'predicted_value': 1 if prediction_result['final_prediction'] == 'FAKE' else 0,
                        'fake_score': prediction_result['aggregate_score'],
                        'detection_rate': prediction_result.get('detection_rate', 0.0),
                        'total_frames': prediction_result.get('total_frames', 0),
                        'detected_frames': prediction_result.get('detected_frames', 0),
                        'prediction_correct': (
                            video_info['true_label'] == prediction_result['final_prediction']
                        )
                    }
                    results.append(result)
                else:
                    print(f"Error processing {video_info['video_id']}: {prediction_result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"Exception processing {video_info['video_id']}: {str(e)}")

        print(f"\nSuccessfully processed {len(results)}/{len(videos)} videos")
        return results

    def calculate_metrics(self, results):
        """Calculate comprehensive metrics for CBAM model evaluation."""
        if not results or len(results) == 0:
            return None

        print("\nCalculating Metrics...")

        # Extract arrays
        true_labels = [r['true_value'] for r in results]
        predicted_labels = [r['predicted_value'] for r in results]
        fake_scores = [r['fake_score'] for r in results]

        # Calculate basic metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)

        # Calculate AUC-ROC
        try:
            auc_roc = roc_auc_score(true_labels, fake_scores)
        except:
            auc_roc = 0.5

        # Calculate per-class metrics
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'confusion_matrix_array': cm.tolist(),
            'total_samples': len(results),
            'correct_predictions': sum(1 for r in results if r['prediction_correct']),
            'processing_statistics': {
                'avg_detection_rate': np.mean([r['detection_rate'] for r in results]),
                'avg_frames_per_video': np.mean([r['total_frames'] for r in results]),
                'avg_detected_faces': np.mean([r['detected_frames'] for r in results])
            }
        }

        # Print summary
        print(f"\nCBAM Model Metrics Summary:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN={tn}, FP={fp}")
        print(f"  FN={fn}, TP={tp}")

        return metrics

    def plot_confusion_matrix(self, metrics):
        """Generate and save confusion matrix visualization."""
        cm = np.array(metrics['confusion_matrix_array'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['REAL', 'FAKE'],
                   yticklabels=['REAL', 'FAKE'])
        plt.title('EfficientNet-CBAM Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cbam_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ CBAM Confusion matrix plot saved")

    def plot_roc_curve(self, results):
        """Generate and save ROC curve."""
        true_labels = [r['true_value'] for r in results]
        fake_scores = [r['fake_score'] for r in results]

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, fake_scores)
        auc_score = roc_auc_score(true_labels, fake_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('EfficientNet-CBAM ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cbam_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ CBAM ROC curve plot saved")

    def save_results(self, results, metrics):
        """Save evaluation results to files."""
        print(f"\nSaving results to: {self.output_dir}")

        # Save metrics
        metrics_path = self.output_dir / "cbam_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save detailed results
        results_path = self.output_dir / "cbam_detailed_results.json"
        serializable_results = []
        for r in results:
            serializable_results.append({
                'video_id': r['video_id'],
                'true_label': r['true_label'],
                'predicted_label': r['predicted_label'],
                'fake_score': r['fake_score'],
                'prediction_correct': r['prediction_correct'],
                'detection_rate': r['detection_rate'],
                'total_frames': r['total_frames'],
                'detected_frames': r['detected_frames']
            })

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        # Save results as CSV
        csv_path = self.output_dir / "cbam_results_summary.csv"
        df_results = pd.DataFrame(results)
        df_results.to_csv(csv_path, index=False)

        # Save plots
        self.plot_confusion_matrix(metrics)
        self.plot_roc_curve(results)

        # Generate summary report
        self.generate_summary_report(results, metrics)

        print(f"✓ All CBAM results saved successfully")

    def generate_summary_report(self, results, metrics):
        """Generate a comprehensive summary report for CBAM model."""
        report = []
        report.append("=" * 80)
        report.append("EFFICIENTNET-CBAM VIDEO-LEVEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"\nModel: EfficientNet-CBAM")
        report.append(f"Checkpoint: {self.model_path}")
        report.append(f"Threshold: {self.threshold}")
        report.append(f"Device: {self.device}")
        report.append(f"\nEvaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Dataset information
        report.append("\n" + "-" * 80)
        report.append("DATASET INFORMATION")
        report.append("-" * 80)
        report.append(f"Dataset Path: {self.dataset_path}")
        report.append(f"Total Videos: {metrics['total_samples']}")
        report.append(f"REAL Videos: {sum(1 for r in results if r['true_label'] == 'REAL')}")
        report.append(f"FAKE Videos: {sum(1 for r in results if r['true_label'] == 'FAKE')}")

        # Performance metrics
        report.append("\n" + "-" * 80)
        report.append("EFFICIENTNET-CBAM PERFORMANCE METRICS")
        report.append("-" * 80)
        report.append(f"Accuracy:      {metrics['accuracy']:.4f}")
        report.append(f"Precision:     {metrics['precision']:.4f}")
        report.append(f"Recall:        {metrics['recall']:.4f}")
        report.append(f"F1-Score:      {metrics['f1_score']:.4f}")
        report.append(f"AUC-ROC:       {metrics['auc_roc']:.4f}")

        # Confusion matrix
        cm = metrics['confusion_matrix']
        report.append("\n" + "-" * 80)
        report.append("CONFUSION MATRIX")
        report.append("-" * 80)
        report.append(f"                    Predicted")
        report.append(f"              REAL        FAKE")
        report.append(f"Actual REAL    {cm['true_negatives']:>8}  {cm['false_positives']:>8}")
        report.append(f"Actual FAKE    {cm['false_negatives']:>8}  {cm['true_positives']:>8}")

        # Processing statistics
        report.append("\n" + "-" * 80)
        report.append("PROCESSING STATISTICS")
        report.append("-" * 80)
        proc_stats = metrics['processing_statistics']
        report.append(f"Average Detection Rate: {proc_stats['avg_detection_rate']:.2%}")
        report.append(f"Average Frames per Video: {proc_stats['avg_frames_per_video']:.1f}")
        report.append(f"Average Detected Faces: {proc_stats['avg_detected_faces']:.1f}")

        # Individual results summary
        report.append("\n" + "-" * 80)
        report.append("PREDICTION RESULTS SUMMARY")
        report.append("-" * 80)
        correct_predictions = metrics['correct_predictions']
        total_predictions = metrics['total_samples']
        report.append(f"Correct Predictions: {correct_predictions}/{total_predictions} ({correct_predictions/total_predictions:.2%})")

        report.append("\n" + "=" * 80)
        report.append("END OF CBAM EVALUATION REPORT")
        report.append("=" * 80)

        # Save report
        report_path = self.output_dir / "cbam_evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

        # Print report to console
        print('\n'.join(report))
        print(f"\n✓ CBAM Summary report saved to: {report_path}")

    def run_evaluation(self):
        """Run complete evaluation for the CBAM model."""
        print("\n" + "="*80)
        print("STARTING EFFICIENTNET-CBAM VIDEO-LEVEL EVALUATION")
        print("="*80)

        # Load dataset
        videos = self.load_video_dataset()
        if not videos:
            print("No videos found in dataset!")
            return None

        # Initialize predictor
        predictor = self.initialize_predictor()

        # Process videos
        results = self.process_videos(videos, predictor)

        if not results:
            print("No results obtained from video processing!")
            return None

        # Calculate metrics
        metrics = self.calculate_metrics(results)

        if metrics:
            # Save results
            self.save_results(results, metrics)

        print("\n" + "="*80)
        print("CBAM EVALUATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nResults saved to: {self.output_dir}")

        return {
            'results': results,
            'metrics': metrics
        }


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate EfficientNet-CBAM model for video-level deepfake detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(CBAM_CHECKPOINT_PATH),
        help="Path to CBAM model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CBAM_CONFIG_PATH),
        help="Path to CBAM configuration JSON file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_PATH,
        help="Path to test-video-level-dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_BASE_DIR),
        help="Base directory for evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        choices=["cuda", "cpu"],
        help="Device for inference"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="Classification threshold"
    )

    args = parser.parse_args()

    print(f"Evaluating EfficientNet-CBAM model")
    print(f"Model checkpoint: {args.model}")
    print(f"Configuration: {args.config}")

    # Initialize evaluator
    evaluator = CBAMModelEvaluator(
        model_path=args.model,
        config_path=args.config,
        dataset_path=args.dataset,
        output_base_dir=args.output,
        threshold=args.threshold,
        device=args.device
    )

    # Run evaluation
    results = evaluator.run_evaluation()

    return results


if __name__ == "__main__":
    results = main()
