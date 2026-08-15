# Aggressive CBAM configuration for better performance
# This config emphasizes stronger attention mechanisms

CBAM_AGGRESSIVE_CONFIG = {
    'model': {
        'name': 'efficientnet_cbam_aggressive',
        'pretrained': True,
        'cbam_variant': 'aggressive',
        'dropout': 0.3,  # Reduced dropout for stronger learning
        'cbam_config': {
            'in_channels': 1280,
            'reduction_ratio': 8,  # More aggressive channel attention
            'kernel_size': 5       # More focused spatial attention
        }
    },
    'training': {
        'epochs': 80,
        'batch_size': 32,
        'learning_rate': 0.00005,  # Lower LR for fine-tuning
        'cbam_lr_multiplier': 3.0,   # Higher LR for CBAM modules
        'classifier_lr_multiplier': 2.0,
        'weight_decay': 0.0001,
        'early_stopping_patience': 15,
        'lr_factor': 0.5,
        'lr_patience': 5,
        'gradient_clip_max_norm': 1.0
    },
    'data': {
        'data_dir': '../preprocessed_dataset',
        'image_size': 224,
        'num_workers': 4
    },
    'transfer_learning': {
        'baseline_path': 'checkpoints/baselines/efficientnet_b0_enhanced_best.pth',
        'freeze_efficientnet': False,  # Fine-tune entire network
        'freeze_epochs': 5             # Unfreeze after 5 epochs
    }
}
