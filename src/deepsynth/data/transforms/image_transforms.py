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
    to_tensor: bool = True,
    # Data augmentation parameters
    use_augmentation: bool = True,
    random_resize_range: Optional[Tuple[int, int]] = None,
    rotation_degrees: float = 3.0,
    perspective_distortion: float = 0.1,
    perspective_prob: float = 0.3,
    color_jitter_brightness: float = 0.1,
    color_jitter_contrast: float = 0.1,
    horizontal_flip_prob: float = 0.3,
) -> transforms.Compose:
    """Create a training transform pipeline with data augmentation for DeepSeek-OCR.

    This pipeline supports both fixed resolution and random augmentation:
    - Fixed mode: Resize to exact resolution (for evaluation/inference)
    - Augmentation mode: Random resize, rotation, perspective, color jitter (for training)

    Args:
        resolution: Base resolution name (tiny/small/base/large/gundam)
        normalize: Whether to normalize pixel values
        normalization_mean: Custom mean values for normalization (default: ImageNet)
        normalization_std: Custom std values for normalization (default: ImageNet)
        to_tensor: Whether to convert to PyTorch tensor
        use_augmentation: Enable random augmentation (recommended for training)
        random_resize_range: (min_size, max_size) for random resize. None = use base resolution ±20%
        rotation_degrees: Max rotation in degrees (±range). 0 = no rotation
        perspective_distortion: Perspective transform strength (0.0-0.5). 0 = disabled
        perspective_prob: Probability of applying perspective transform
        color_jitter_brightness: Brightness variation (0.0-1.0). 0 = disabled
        color_jitter_contrast: Contrast variation (0.0-1.0). 0 = disabled
        horizontal_flip_prob: Probability of horizontal flip (0.0-1.0)

    Returns:
        Composed transform pipeline

    Examples:
        >>> # Training with augmentation (default)
        >>> transform = create_training_transform(resolution="base", use_augmentation=True)

        >>> # Evaluation without augmentation
        >>> transform = create_training_transform(resolution="base", use_augmentation=False)

        >>> # Custom augmentation parameters
        >>> transform = create_training_transform(
        ...     resolution="base",
        ...     random_resize_range=(800, 1200),
        ...     rotation_degrees=5.0,
        ...     perspective_distortion=0.15
        ... )

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

    transform_list = []

    if use_augmentation:
        # Random augmentation pipeline for training
        base_size = DEEPSEEK_OCR_RESOLUTIONS[resolution]

        # Determine random resize range
        if random_resize_range is None:
            # Default: ±20% of base resolution
            min_size = int(base_size[0] * 0.8)
            max_size = int(base_size[0] * 1.2)
        else:
            min_size, max_size = random_resize_range

        # RandomResizedCrop with scale variation
        transform_list.append(
            transforms.RandomResizedCrop(
                size=base_size,
                scale=(min_size / base_size[0], max_size / base_size[0]),
                ratio=(0.9, 1.1),  # Slight aspect ratio variation
                interpolation=Image.Resampling.LANCZOS
            )
        )

        # Random rotation
        if rotation_degrees > 0:
            transform_list.append(
                transforms.RandomRotation(
                    degrees=(-rotation_degrees, rotation_degrees),
                    interpolation=Image.Resampling.BILINEAR,
                    fill=255  # White fill for rotated areas
                )
            )

        # Random perspective
        if perspective_distortion > 0 and perspective_prob > 0:
            transform_list.append(
                transforms.RandomPerspective(
                    distortion_scale=perspective_distortion,
                    p=perspective_prob,
                    interpolation=Image.Resampling.BILINEAR,
                    fill=255
                )
            )

        # Color jitter
        if color_jitter_brightness > 0 or color_jitter_contrast > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=color_jitter_brightness,
                    contrast=color_jitter_contrast
                )
            )

        # Random horizontal flip
        if horizontal_flip_prob > 0:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob)
            )
    else:
        # Fixed resize for evaluation/inference
        transform_list.append(ResizeTransform(resolution, padding=True))

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

    For inference, we don't use augmentation - just fixed resize.
    This ensures consistent results.

    Args:
        resolution: Target resolution (tiny/small/base/large/gundam)

    Returns:
        Composed transform pipeline for inference
    """
    return create_training_transform(
        resolution=resolution,
        use_augmentation=False,  # No random augmentation for inference
        normalize=True,
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
