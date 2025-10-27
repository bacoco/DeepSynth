"""Image transformation pipeline for training.

This module provides standard transforms for resizing and preprocessing
images during training. Images are resized on-the-fly during batch loading
using HuggingFace dataset transforms.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from PIL import Image

try:
    from torchvision import transforms
except ImportError:
    transforms = None

from .text_to_image import DEEPSEEK_OCR_RESOLUTIONS


class ResizeTransform:
    """Resize images to target resolution with optional padding.

    This transform maintains aspect ratio and adds padding to reach the exact
    target size, ensuring no distortion of the image content.

    Args:
        target_resolution: Resolution name (tiny/small/base/large/gundam) or custom (width, height) tuple
        padding: If True, preserve aspect ratio and pad. If False, resize directly (may distort).
        background_color: RGB tuple for padding color (default: white)
    """

    def __init__(
        self,
        target_resolution: str | Tuple[int, int] = "base",
        padding: bool = True,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ):
        if isinstance(target_resolution, str):
            if target_resolution not in DEEPSEEK_OCR_RESOLUTIONS:
                raise ValueError(
                    f"Unknown resolution '{target_resolution}'. "
                    f"Choose from: {list(DEEPSEEK_OCR_RESOLUTIONS.keys())}"
                )
            self.target_size = DEEPSEEK_OCR_RESOLUTIONS[target_resolution]
        else:
            self.target_size = target_resolution

        self.padding = padding
        self.background_color = background_color

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply the resize transform to an image.

        Args:
            image: PIL Image to resize

        Returns:
            Resized PIL Image
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image, got {type(image)}")

        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.padding:
            return self._resize_with_padding(image)
        else:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)

    def _resize_with_padding(self, image: Image.Image) -> Image.Image:
        """Resize image preserving aspect ratio with padding to reach target size.

        This is the same logic as in text_to_image.py to ensure consistency.
        """
        target_width, target_height = self.target_size
        orig_width, orig_height = image.size

        # Calculate scaling factor to fit within target while preserving aspect ratio
        scale = min(target_width / orig_width, target_height / orig_height)

        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Resize image using high-quality LANCZOS filter
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create new image with target size and background color
        result = Image.new("RGB", self.target_size, color=self.background_color)

        # Calculate position to paste resized image (center it)
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        # Paste resized image onto padded background
        result.paste(resized, (paste_x, paste_y))

        return result


def create_training_transform(
    resolution: str = "base",
    normalize: bool = True,
    normalization_mean: Optional[Tuple[float, float, float]] = None,
    normalization_std: Optional[Tuple[float, float, float]] = None,
    to_tensor: bool = True
) -> transforms.Compose:
    """Create a standard training transform pipeline for DeepSeek-OCR.

    This pipeline:
    1. Resizes images to target resolution with padding
    2. Converts to tensor (optional)
    3. Normalizes pixel values (optional)

    Args:
        resolution: Target resolution (tiny/small/base/large/gundam)
        normalize: Whether to normalize pixel values
        normalization_mean: Custom mean values for normalization (default: ImageNet)
        normalization_std: Custom std values for normalization (default: ImageNet)
        to_tensor: Whether to convert to PyTorch tensor

    Returns:
        Composed transform pipeline

    Example:
        >>> transform = create_training_transform(resolution="base")
        >>> # Apply to HuggingFace dataset
        >>> def apply_transform(examples):
        ...     examples['pixel_values'] = [transform(img) for img in examples['image']]
        ...     return examples
        >>> dataset.set_transform(apply_transform)
    """
    if transforms is None:
        raise RuntimeError(
            "torchvision is required for training transforms. "
            "Install with `pip install torchvision`"
        )

    transform_list = [
        ResizeTransform(resolution, padding=True),
    ]

    if to_tensor:
        transform_list.append(transforms.ToTensor())

    if normalize:
        # Default to ImageNet normalization (standard for vision models)
        mean = normalization_mean or (0.485, 0.456, 0.406)
        std = normalization_std or (0.229, 0.224, 0.225)
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


def create_inference_transform(resolution: str = "base") -> transforms.Compose:
    """Create a minimal inference transform pipeline.

    For inference, we typically don't need normalization (depends on model).
    This just resizes and converts to tensor.

    Args:
        resolution: Target resolution (tiny/small/base/large/gundam)

    Returns:
        Composed transform pipeline for inference
    """
    return create_training_transform(
        resolution=resolution,
        normalize=False,
        to_tensor=True
    )


def apply_transform_to_dataset(dataset, transform_fn, image_column: str = "image"):
    """Apply a transform function to a HuggingFace dataset.

    This is a helper function that sets up the transform to be applied
    on-the-fly during batch loading.

    Args:
        dataset: HuggingFace Dataset object
        transform_fn: Transform function or pipeline to apply
        image_column: Name of the image column (default: "image")

    Returns:
        Dataset with transform applied

    Example:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("repo/dataset")
        >>> transform = create_training_transform("base")
        >>> dataset = apply_transform_to_dataset(dataset, transform)
    """
    def _apply_transform(examples):
        # Apply transform to all images in the batch
        examples['pixel_values'] = [
            transform_fn(img) for img in examples[image_column]
        ]
        return examples

    dataset.set_transform(_apply_transform)
    return dataset


# Export public API
__all__ = [
    "ResizeTransform",
    "create_training_transform",
    "create_inference_transform",
    "apply_transform_to_dataset",
    "DEEPSEEK_OCR_RESOLUTIONS",
]
