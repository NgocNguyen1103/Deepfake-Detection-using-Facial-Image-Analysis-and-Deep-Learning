# Deepfake Detection System

A complete deepfake detection pipeline using EfficientNet-B0 with face detection and comprehensive evaluation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess data
python scripts/run_preprocess.py

# Train baseline model
python training/train_baseline.py

# Evaluate model
python scripts/evaluate_baseline_model_image_level.py

# Predict on videos
python inference/video_predictor.py --checkpoint checkpoints/baselines/efficientnet_b0_enhanced_best.pth

# Run the Streamlit research demo
pip install -r requirements-demo.txt
streamlit run app.py
```

## Project Structure

```
Project Code/
├── models/              # Model architectures
├── training/            # Training scripts
├── inference/           # Prediction scripts
├── utils/              # Utility functions
├── configs/            # Configuration files
├── docs/               # Documentation
├── scripts/            # Executable scripts
├── data/               # Data loading
├── preprocessing/      # Preprocessing pipeline
├── checkpoints/        # Model weights
└── results/            # Evaluation results
```

## Documentation

- [Detailed README](docs/README.md)
- [Evaluation Guide](docs/EVALUATION_GUIDE.md)
- [Pipeline Guide](docs/VIDEO_PIPELINE_GUIDE.md)
- [Anti-Overfitting Guide](docs/ANTI_OVERFITTING_GUIDE.md)

## Model Performance

- **Baseline Model**: EfficientNet-B0 with ImageNet pretrained weights
- **Enhanced Model**: Added anti-overfitting techniques (early stopping, learning rate scheduling, mixup augmentation)
- **Accuracy**: ~82% on test set
- **F1-Score**: ~0.83

## License

MIT License
