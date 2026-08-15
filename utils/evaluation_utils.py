# evaluation_utils.py
# Utility functions for video-level evaluation analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score
import json


def create_score_distribution_plot(results, output_path, title="Score Distribution"):
    """Create distribution plot of fake scores by true label."""
    real_scores = [r['fake_score'] for r in results if r['true_label'] == 'REAL']
    fake_scores = [r['fake_score'] for r in results if r['true_label'] == 'FAKE']

    plt.figure(figsize=(10, 6))
    plt.hist(real_scores, bins=30, alpha=0.7, label='REAL', color='green')
    plt.hist(fake_scores, bins=30, alpha=0.7, label='FAKE', color='red')
    plt.xlabel('Fake Score')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_precision_recall_curve(results, output_path):
    """Create Precision-Recall curve."""
    true_labels = [r['true_value'] for r in results]
    fake_scores = [r['fake_score'] for r in results]

    precision, recall, thresholds = precision_recall_curve(true_labels, fake_scores)
    avg_precision = average_precision_score(true_labels, fake_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_error_analysis_plot(results, output_path):
    """Create error analysis visualization."""
    errors = [r for r in results if not r['prediction_correct']]

    if not errors:
        print("No errors to analyze")
        return

    error_types = {}
    for error in errors:
        true_label = error['true_label']
        predicted_label = error['predicted_label']
        error_type = f"{true_label} predicted as {predicted_label}"
        error_types[error_type] = error_types.get(error_type, 0) + 1

    plt.figure(figsize=(8, 6))
    error_types_keys = list(error_types.keys())
    error_types_values = list(error_types.values())

    colors = ['red', 'orange'] if len(error_types_keys) == 2 else None
    plt.bar(error_types_keys, error_types_values, color=colors)
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    plt.title('Error Analysis')
    plt.xticks(rotation=15)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def load_evaluation_results(results_dir):
    """Load evaluation results from a directory."""
    results_path = Path(results_dir)

    with open(results_path / 'metrics.json', 'r') as f:
        metrics = json.load(f)

    with open(results_path / 'detailed_results.json', 'r') as f:
        detailed_results = json.load(f)

    return {
        'metrics': metrics,
        'detailed_results': detailed_results
    }


def compare_models(results_dirs, output_path):
    """Compare multiple models and create comparison visualizations."""
    comparison_data = []

    for model_dir in results_dirs:
        model_name = Path(model_dir).name
        results = load_evaluation_results(model_dir)

        comparison_data.append({
            'model': model_name,
            'accuracy': results['metrics']['accuracy'],
            'precision': results['metrics']['precision'],
            'recall': results['metrics']['recall'],
            'f1_score': results['metrics']['f1_score'],
            'auc_roc': results['metrics']['auc_roc']
        })

    df = pd.DataFrame(comparison_data)

    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        df.plot(x='model', y=metric, kind='bar', ax=ax, legend=False)
        ax.set_title(metric.upper(), fontsize=10)
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

    # Remove empty subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_performance_summary(results_dir, output_path):
    """Create a performance summary visualization."""
    results = load_evaluation_results(results_dir)
    metrics = results['metrics']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix heatmap
    cm_array = np.array(metrics['confusion_matrix_array'])
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
               xticklabels=['REAL', 'FAKE'],
               yticklabels=['REAL', 'FAKE'],
               ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Performance metrics radar chart
    metrics_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['auc_roc']
    ]

    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

    # Create simple bar chart instead of radar
    axes[1].bar(categories, metrics_values, color='steelblue')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Performance Metrics')
    axes[1].set_ylabel('Score')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_latex_table(results_dirs, output_path):
    """Generate LaTeX table for model comparison."""
    comparison_data = []

    for model_dir in results_dirs:
        model_name = Path(model_dir).name.replace('_', ' ').title()
        results = load_evaluation_results(model_dir)

        comparison_data.append({
            'model': model_name,
            'accuracy': results['metrics']['accuracy'],
            'precision': results['metrics']['precision'],
            'recall': results['metrics']['recall'],
            'f1_score': results['metrics']['f1_score'],
            'auc_roc': results['metrics']['auc_roc']
        })

    df = pd.DataFrame(comparison_data)

    latex = df.to_latex(index=False, float_format='%.4f')

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"LaTeX table saved to {output_path}")


def calculate_per_video_type_metrics(results, output_path):
    """Calculate metrics grouped by video characteristics."""
    # Group by manipulation type if available
    detailed_df = pd.DataFrame(results)

    if 'manipulation_type' in detailed_df.columns:
        manipulation_metrics = {}

        for manipulation_type in detailed_df['manipulation_type'].unique():
            if pd.notna(manipulation_type):
                type_subset = detailed_df[detailed_df['manipulation_type'] == manipulation_type]

                if len(type_subset) > 0:
                    correct = sum(type_subset['prediction_correct'])
                    total = len(type_subset)
                    accuracy = correct / total if total > 0 else 0

                    manipulation_metrics[manipulation_type] = {
                        'accuracy': accuracy,
                        'total_videos': total,
                        'correct_predictions': correct
                    }

        # Save manipulation type metrics
        with open(output_path, 'w') as f:
            json.dump(manipulation_metrics, f, indent=2)

        return manipulation_metrics

    return None


def create_threshold_analysis(results, output_path, thresholds=np.arange(0.1, 0.9, 0.05)):
    """Analyze model performance across different thresholds."""
    true_labels = np.array([r['true_value'] for r in results])
    fake_scores = np.array([r['fake_score'] for r in results])

    threshold_results = []

    for threshold in thresholds:
        predictions = (fake_scores >= threshold).astype(int)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)

        threshold_results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

    # Plot threshold analysis
    df_thresholds = pd.DataFrame(threshold_results)

    plt.figure(figsize=(10, 6))
    plt.plot(df_thresholds['threshold'], df_thresholds['accuracy'], marker='o', label='Accuracy')
    plt.plot(df_thresholds['threshold'], df_thresholds['precision'], marker='s', label='Precision')
    plt.plot(df_thresholds['threshold'], df_thresholds['recall'], marker='^', label='Recall')
    plt.plot(df_thresholds['threshold'], df_thresholds['f1_score'], marker='d', label='F1-Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return threshold_results


def save_evaluation_summary(evaluation_results, output_path):
    """Save a comprehensive evaluation summary."""
    summary = {
        'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models_evaluated': list(evaluation_results.keys()),
        'summary': {}
    }

    for model_key, model_data in evaluation_results.items():
        metrics = model_data['metrics']
        summary['summary'][model_key] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'auc_roc': metrics['auc_roc'],
            'total_samples': metrics['total_samples'],
            'correct_predictions': metrics['correct_predictions']
        }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluation summary saved to {output_path}")


if __name__ == "__main__":
    print("Evaluation utilities loaded successfully")
    print("Available functions:")
    print("- create_score_distribution_plot")
    print("- create_precision_recall_curve")
    print("- create_error_analysis_plot")
    print("- compare_models")
    print("- create_performance_summary")
    print("- generate_latex_table")
    print("- create_threshold_analysis")