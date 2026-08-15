# evaluate_video_dataset.py
# Single model evaluation for video-level deepfake detection

import os
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
from video_predictor import VideoPredictor

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# ====================================================================
# MODEL CONFIGURATION - CHANGE THIS TO SELECT WHICH MODEL TO EVALUATE
# ====================================================================

# Options: 'trained' or 'pretrained'
MODEL_TO_EVALUATE = 'pretrained'

# Model configurations
MODELS_AVAILABLE = {
    'trained': {
        'name': 'EfficientNet-B0-Trained',
        'description': 'EfficientNet-B0 with trained weights from deepfake dataset',
        'model_path': 'checkpoints/efficientnet_b0_enhanced_best.pth',
        'use_pretrained': False,
        'output_folder': 'efficientnet_b0_trained'
    },
    'pretrained': {
        'name': 'EfficientNet-B0-Pretrained',
        'description': 'EfficientNet-B0 with ImageNet pretrained weights',
        'model_path': None,
        'use_pretrained': True,
        'output_folder': 'efficientnet_b0_pretrained'
    }
}

# ====================================================================
# EVALUATION SETTINGS
# ====================================================================

DATASET_PATH = "D:/M1/M1 Internship/test-video-level-dataset"
OUTPUT_BASE_DIR = "evaluation_results"
DEVICE = "cuda"  # Options: "cuda" or "cpu"
THRESHOLD = 0.5

# ====================================================================


class SingleModelEvaluator:
    """
    Evaluate a single video-level deepfake detection model on test dataset.

    Evaluates one model at a time with comprehensive metrics and analysis.
    """

    def __init__(
        self,
        model_key: str,
        dataset_path: str,
        output_base_dir: str = "evaluation_results",
        threshold: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize the evaluator for a single model.

        Args:
            model_key: Key from MODELS_AVAILABLE dict ('trained' or 'pretrained')
            dataset_path: Path to test-video-level-dataset
            output_base_dir: Base directory for evaluation results
            threshold: Classification threshold
            device: Device for inference (cuda/cpu)
        """
        if model_key not in MODELS_AVAILABLE:
            raise ValueError(f"Model key '{model_key}' not found. Available: {list(MODELS_AVAILABLE.keys())}")

        self.model_config = MODELS_AVAILABLE[model_key]
        self.model_key = model_key
        self.dataset_path = Path(dataset_path)
        self.output_base_dir = Path(output_base_dir)
        self.threshold = threshold
        self.device = device

        # Create model-specific output directory
        self.output_dir = self.output_base_dir / self.model_config['output_folder']
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Single Model Evaluator initialized")
        print(f"Model: {self.model_config['name']}")
        print(f"Description: {self.model_config['description']}")
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
        """Initialize video predictor for the selected model."""
        print("\n" + "="*60)
        print("Initializing Model Predictor")
        print("="*60)

        try:
            if self.model_config['use_pretrained']:
                # For pretrained model
                from model.efficientnet_baseline import EfficientNetB0Baseline

                predictor = VideoPredictor(
                    model_path=None,
                    device=self.device,
                    threshold=self.threshold,
                    frame_interval=1
                )

                # Load pretrained model
                model = EfficientNetB0Baseline(pretrained=True)
                model = model.to(predictor.device)
                model.eval()
                predictor.model = model

                print(f"✓ Loaded ImageNet pretrained weights")

            else:
                # Use trained weights
                predictor = VideoPredictor(
                    model_path=self.model_config['model_path'],
                    device=self.device,
                    threshold=self.threshold,
                    frame_interval=1
                )
                print(f"✓ Loaded trained weights from: {self.model_config['model_path']}")

            return predictor

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def process_videos(self, videos, predictor):
        """Process all videos with the model."""
        print(f"\n{'='*60}")
        print(f"Processing videos with {self.model_config['name']}")
        print(f"{'='*60}")

        results = []

        for video_info in tqdm(videos, desc=f"Processing"):
            video_path = video_info['video_path']

            try:
                # Get prediction
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
        """Calculate comprehensive metrics for model evaluation."""
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
        print(f"\nMetrics Summary:")
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
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrix plot saved")

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
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ ROC curve plot saved")

    def save_results(self, results, metrics):
        """Save evaluation results to files."""
        print(f"\nSaving results to: {self.output_dir}")

        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save detailed results
        results_path = self.output_dir / "detailed_results.json"
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
        csv_path = self.output_dir / "results_summary.csv"
        df_results = pd.DataFrame(results)
        df_results.to_csv(csv_path, index=False)

        # Save plots
        self.plot_confusion_matrix(metrics)
        self.plot_roc_curve(results)

        # Generate summary report
        self.generate_summary_report(results, metrics)

        print(f"✓ All results saved successfully")

    def generate_summary_report(self, results, metrics):
        """Generate a comprehensive summary report."""
        report = []
        report.append("=" * 80)
        report.append(f"VIDEO-LEVEL DEEPFAKE DETECTION EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"\nModel: {self.model_config['name']}")
        report.append(f"Description: {self.model_config['description']}")
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
        report.append("PERFORMANCE METRICS")
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
        report.append("END OF REPORT")
        report.append("=" * 80)

        # Save report
        report_path = self.output_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

        # Print report to console
        print('\n'.join(report))
        print(f"\n✓ Summary report saved to: {report_path}")

    def run_evaluation(self):
        """Run complete evaluation for the selected model."""
        print("\n" + "="*80)
        print(f"STARTING SINGLE MODEL EVALUATION: {self.model_config['name']}")
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
        print("EVALUATION COMPLETED SUCCESSFULLY")
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
        description="Evaluate single video-level deepfake detection model"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=['trained', 'pretrained'],
        default=None,
        help="Model to evaluate (trained/pretrained). If not specified, uses MODEL_TO_EVALUATE from code."
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
        default=OUTPUT_BASE_DIR,
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

    # Determine which model to evaluate
    model_to_evaluate = args.model if args.model else MODEL_TO_EVALUATE

    print(f"Evaluating model: {model_to_evaluate}")
    print(f"Model configuration: {MODELS_AVAILABLE[model_to_evaluate]['name']}")

    # Initialize evaluator
    evaluator = SingleModelEvaluator(
        model_key=model_to_evaluate,
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
