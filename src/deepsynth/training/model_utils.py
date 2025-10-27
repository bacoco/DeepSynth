"""
Model utilities for DeepSynth training.
Provides robust parameter freezing and model introspection.
"""

import logging
from typing import Dict, List, Tuple

import torch
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def freeze_vision_encoder(model: PreTrainedModel) -> Dict[str, int]:
    """
    Freeze vision encoder parameters explicitly by known module names.

    Args:
        model: HuggingFace model (e.g., DeepSeek-OCR)

    Returns:
        Dictionary with counts of frozen/trainable parameters
    """
    # Known vision encoder module patterns for various architectures
    vision_module_patterns = [
        'vision_encoder',
        'vision_tower',
        'visual_encoder',
        'vision_model',
        'vision',
        'visual',
        'encoder.layers',  # Generic encoder layers
        'vit',  # Vision Transformer
        'sam',  # Segment Anything
        'clip',  # CLIP encoder
    ]

    frozen_params = 0
    frozen_modules = []

    # Freeze by explicit module attributes first
    for attr_name in ['vision_encoder', 'vision_tower', 'visual_encoder', 'vision_model']:
        if hasattr(model, attr_name):
            module = getattr(model, attr_name)
            for param in module.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            frozen_modules.append(attr_name)
            logger.info(f"✓ Frozen {attr_name} module")

    # If no explicit vision module found, use pattern matching as fallback
    if not frozen_modules:
        logger.warning("No explicit vision encoder found, using pattern matching")
        for name, param in model.named_parameters():
            name_lower = name.lower()
            if any(pattern in name_lower for pattern in vision_module_patterns):
                param.requires_grad = False
                frozen_params += param.numel()
                if name not in frozen_modules:
                    frozen_modules.append(name.split('.')[0])

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    stats = {
        'frozen_params': frozen_params,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'frozen_modules': frozen_modules,
        'frozen_percentage': (frozen_params / total_params * 100) if total_params > 0 else 0,
    }

    # Validation: ensure we froze something
    if frozen_params == 0:
        logger.error("❌ WARNING: No vision encoder parameters were frozen!")
        logger.error("Model may not follow expected architecture")
        logger.error("Available modules: " + ", ".join([n.split('.')[0] for n, _ in model.named_modules()]))
    else:
        logger.info(f"✓ Froze {frozen_params:,} parameters ({stats['frozen_percentage']:.1f}%)")
        logger.info(f"✓ Trainable parameters: {trainable_params:,} ({100 - stats['frozen_percentage']:.1f}%)")
        logger.info(f"✓ Frozen modules: {', '.join(frozen_modules)}")

    return stats


def get_parameter_groups(model: PreTrainedModel) -> List[Tuple[str, int, bool]]:
    """
    Analyze model parameter groups and their freeze status.

    Args:
        model: HuggingFace model

    Returns:
        List of (module_name, param_count, is_frozen) tuples
    """
    param_groups = {}

    for name, param in model.named_parameters():
        # Extract top-level module name
        module_name = name.split('.')[0]

        if module_name not in param_groups:
            param_groups[module_name] = {
                'params': 0,
                'frozen': 0,
                'trainable': 0
            }

        param_count = param.numel()
        param_groups[module_name]['params'] += param_count

        if param.requires_grad:
            param_groups[module_name]['trainable'] += param_count
        else:
            param_groups[module_name]['frozen'] += param_count

    # Convert to list of tuples
    result = []
    for module_name, counts in param_groups.items():
        is_frozen = counts['frozen'] == counts['params']
        result.append((module_name, counts['params'], is_frozen))

    # Sort by parameter count (descending)
    result.sort(key=lambda x: x[1], reverse=True)

    return result


def print_parameter_summary(model: PreTrainedModel) -> None:
    """
    Print a detailed summary of model parameters and freeze status.

    Args:
        model: HuggingFace model
    """
    logger.info("=" * 80)
    logger.info("MODEL PARAMETER SUMMARY")
    logger.info("=" * 80)

    param_groups = get_parameter_groups(model)

    logger.info(f"{'Module':<30} {'Params':<15} {'Status'}")
    logger.info("-" * 80)

    for module_name, param_count, is_frozen in param_groups:
        status = "🔒 FROZEN" if is_frozen else "✏️  TRAINABLE"
        logger.info(f"{module_name:<30} {param_count:>13,}  {status}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info("-" * 80)
    logger.info(f"{'TOTAL':<30} {total_params:>13,}")
    logger.info(f"{'Trainable':<30} {trainable_params:>13,}  ({trainable_params/total_params*100:.1f}%)")
    logger.info(f"{'Frozen':<30} {frozen_params:>13,}  ({frozen_params/total_params*100:.1f}%)")
    logger.info("=" * 80)


def validate_vision_model(model: PreTrainedModel) -> bool:
    """
    Validate that the model has vision capabilities.

    Args:
        model: HuggingFace model

    Returns:
        True if vision encoder detected, False otherwise
    """
    vision_attrs = ['vision_encoder', 'vision_tower', 'visual_encoder', 'vision_model']

    for attr in vision_attrs:
        if hasattr(model, attr):
            logger.info(f"✓ Vision model detected: {attr}")
            return True

    # Check for vision-related modules in the model
    vision_modules = [
        name for name, _ in model.named_modules()
        if any(pattern in name.lower() for pattern in ['vision', 'visual', 'vit', 'image'])
    ]

    if vision_modules:
        logger.info(f"✓ Vision modules detected: {', '.join(vision_modules[:5])}")
        return True

    logger.warning("⚠️ No vision encoder detected in model")
    logger.warning("This model may not support vision-to-text training")
    return False


def freeze_embeddings(model: PreTrainedModel) -> int:
    """
    Freeze embedding layers to prevent catastrophic forgetting.

    Args:
        model: HuggingFace model

    Returns:
        Number of parameters frozen
    """
    frozen_count = 0

    for name, param in model.named_parameters():
        if 'embed' in name.lower():
            param.requires_grad = False
            frozen_count += param.numel()

    if frozen_count > 0:
        logger.info(f"✓ Froze {frozen_count:,} embedding parameters")

    return frozen_count
