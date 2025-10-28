"""Transformation helpers for dataset preparation."""
from .text_to_image import DEEPSEEK_OCR_RESOLUTIONS, TextToImageConverter, batch_convert

# Lazy import for image_transforms to avoid loading torchvision
# when only using text_to_image utilities
def __getattr__(name):
    if name in ("ResizeTransform", "create_training_transform",
                "create_inference_transform", "apply_transform_to_dataset"):
        from .image_transforms import (
            ResizeTransform,
            create_training_transform,
            create_inference_transform,
            apply_transform_to_dataset,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DEEPSEEK_OCR_RESOLUTIONS",
    "TextToImageConverter",
    "batch_convert",
    "ResizeTransform",
    "create_training_transform",
    "create_inference_transform",
    "apply_transform_to_dataset",
]
