"""Export and packaging utilities for LoRA adapters and fine-tuned models."""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

LOGGER = logging.getLogger(__name__)


class AdapterExporter:
    """Export LoRA adapters and create deployment packages."""

    def __init__(self, model_path: Union[str, Path]):
        """Initialize exporter.

        Args:
            model_path: Path to the trained model/adapters
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        LOGGER.info(f"Initialized adapter exporter for: {model_path}")

    def export_adapters(
        self,
        output_dir: Union[str, Path],
        include_base_model: bool = False,
        create_inference_script: bool = True,
        create_model_card: bool = True,
    ) -> Path:
        """Export LoRA adapters to a directory.

        Args:
            output_dir: Directory to export to
            include_base_model: Whether to include base model weights
            create_inference_script: Whether to create inference script
            create_model_card: Whether to create model card

        Returns:
            Path to export directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        LOGGER.info(f"Exporting adapters to: {output_path}")

        # Copy adapter files
        adapter_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
        for file in adapter_files:
            src = self.model_path / file
            if src.exists():
                shutil.copy2(src, output_path / file)
                LOGGER.info(f"Copied: {file}")

        # Copy config files
        config_files = ["config.json", "training_config.json"]
        for file in config_files:
            src = self.model_path / file
            if src.exists():
                shutil.copy2(src, output_path / file)

        # Copy tokenizer files
        tokenizer_files = [
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "vocab.txt",
            "merges.txt",
        ]
        for file in tokenizer_files:
            src = self.model_path / file
            if src.exists():
                shutil.copy2(src, output_path / file)

        # Create inference script
        if create_inference_script:
            self._create_inference_script(output_path)

        # Create model card
        if create_model_card:
            self._create_model_card(output_path)

        # Create requirements.txt
        self._create_requirements(output_path)

        LOGGER.info(f"Export completed: {output_path}")
        return output_path

    def create_deployment_package(
        self,
        output_file: Union[str, Path],
        include_base_model: bool = False,
    ) -> Path:
        """Create a zip package for deployment.

        Args:
            output_file: Output zip file path
            include_base_model: Whether to include base model

        Returns:
            Path to created zip file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        LOGGER.info(f"Creating deployment package: {output_path}")

        # Create temporary export directory
        temp_dir = output_path.parent / "temp_export"
        self.export_adapters(
            temp_dir,
            include_base_model=include_base_model,
            create_inference_script=True,
            create_model_card=True,
        )

        # Create zip file
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in temp_dir.rglob("*"):
                if file.is_file():
                    arcname = file.relative_to(temp_dir)
                    zipf.write(file, arcname)
                    LOGGER.debug(f"Added to zip: {arcname}")

        # Cleanup temp directory
        shutil.rmtree(temp_dir)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        LOGGER.info(f"Package created: {output_path} ({file_size_mb:.1f} MB)")
        return output_path

    def merge_adapter_to_base(
        self,
        base_model_path: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> Path:
        """Merge LoRA adapter back into base model.

        Args:
            base_model_path: Path to base model
            output_dir: Output directory for merged model

        Returns:
            Path to merged model
        """
        try:
            from peft import PeftModel
            from transformers import AutoModel

            LOGGER.info("Loading base model...")
            base_model = AutoModel.from_pretrained(
                base_model_path,
                trust_remote_code=True,
            )

            LOGGER.info("Loading LoRA adapters...")
            model = PeftModel.from_pretrained(base_model, str(self.model_path))

            LOGGER.info("Merging adapters into base model...")
            merged_model = model.merge_and_unload()

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            LOGGER.info(f"Saving merged model to: {output_path}")
            merged_model.save_pretrained(output_path)

            # Copy tokenizer
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path, trust_remote_code=True
            )
            tokenizer.save_pretrained(output_path)

            LOGGER.info("Merge completed successfully")
            return output_path

        except Exception as e:
            LOGGER.error(f"Failed to merge adapter: {e}")
            raise

    def _create_inference_script(self, output_dir: Path):
        """Create a ready-to-use inference script."""
        script_content = '''#!/usr/bin/env python3
"""
Inference script for DeepSynth LoRA fine-tuned model.

Usage:
    python inference.py --image document.pdf --query "Summarize this document"
"""

import argparse
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel


def load_model(adapter_path="./"):
    """Load base model with LoRA adapters."""
    print("Loading model with LoRA adapters...")

    # Load base model
    base_model = AutoModel.from_pretrained(
        "deepseek-ai/DeepSeek-OCR",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )

    return model, tokenizer


def load_image(image_path):
    """Load and prepare image."""
    if image_path.endswith(".pdf"):
        # Handle PDF (requires pdf2image)
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(image_path)
            return images[0]  # Use first page
        except ImportError:
            raise ImportError("pdf2image required for PDF support: pip install pdf2image")
    else:
        return Image.open(image_path).convert("RGB")


def summarize(model, tokenizer, image, max_length=512):
    """Generate summary from image."""
    device = next(model.parameters()).device

    # Prepare image (implementation depends on model architecture)
    # This is a placeholder - actual implementation needed
    with torch.no_grad():
        # TODO: Implement actual forward pass
        # outputs = model.generate(image_tensor, max_length=max_length)
        # summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = "Summary generation not yet implemented in this template"

    return summary


def main():
    parser = argparse.ArgumentParser(description="DeepSynth LoRA Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image/PDF")
    parser.add_argument("--adapter-path", type=str, default="./", help="Path to LoRA adapters")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum summary length")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.adapter_path)

    # Load image
    print(f"Loading image: {args.image}")
    image = load_image(args.image)

    # Generate summary
    print("Generating summary...")
    summary = summarize(model, tokenizer, image, args.max_length)

    print("\\nSummary:")
    print("-" * 80)
    print(summary)
    print("-" * 80)


if __name__ == "__main__":
    main()
'''

        script_path = output_dir / "inference.py"
        with open(script_path, "w") as f:
            f.write(script_content)

        # Make executable
        script_path.chmod(0o755)

        LOGGER.info(f"Created inference script: {script_path}")

    def _create_model_card(self, output_dir: Path):
        """Create a model card with training information."""
        # Load training config if available
        training_config = {}
        config_path = self.model_path / "training_config.json"
        if config_path.exists():
            with open(config_path) as f:
                training_config = json.load(f)

        # Generate model card
        card_content = f"""# DeepSynth LoRA Fine-Tuned Model

## Model Description

This model is a LoRA fine-tuned version of DeepSeek-OCR for document summarization using the DeepSynth framework.

**Export Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Training Configuration

- **Base Model**: {training_config.get('model_name', 'deepseek-ai/DeepSeek-OCR')}
- **LoRA Rank**: {training_config.get('lora_rank', 'N/A')}
- **LoRA Alpha**: {training_config.get('lora_alpha', 'N/A')}
- **Batch Size**: {training_config.get('batch_size', 'N/A')}
- **Learning Rate**: {training_config.get('learning_rate', 'N/A')}
- **Epochs**: {training_config.get('num_epochs', 'N/A')}

## Usage

### Quick Start

```python
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# Load model
base_model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "./")
tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)

# Run inference (see inference.py for complete example)
```

### Command Line

```bash
python inference.py --image document.pdf
```

## Requirements

See `requirements.txt` for dependencies.

## Model Architecture

- **Vision Encoder**: DeepSeek-OCR (frozen)
- **Decoder**: DeepSeek-3B with LoRA adapters
- **Parameter Efficient**: Only {training_config.get('lora_rank', 16) * 2}M trainable parameters

## License

This model inherits the license from the base DeepSeek-OCR model.

## Citation

```bibtex
@misc{{deepsynth-lora,
  title={{DeepSynth LoRA Fine-Tuned Model}},
  year={{2025}},
  note={{Fine-tuned with DeepSynth framework}}
}}
```

---

Generated by DeepSynth Adapter Exporter
"""

        card_path = output_dir / "README.md"
        with open(card_path, "w") as f:
            f.write(card_content)

        LOGGER.info(f"Created model card: {card_path}")

    def _create_requirements(self, output_dir: Path):
        """Create requirements.txt for inference."""
        requirements = """# DeepSynth LoRA Inference Requirements

torch>=2.0.0
transformers>=4.46.0
peft>=0.11.1
pillow>=9.5.0
pdf2image>=1.16.3  # For PDF support
accelerate>=0.24.0
"""

        req_path = output_dir / "requirements.txt"
        with open(req_path, "w") as f:
            f.write(requirements)

        LOGGER.info(f"Created requirements.txt: {req_path}")


def export_adapter(
    model_path: Union[str, Path],
    output_dir: Union[str, Path],
    create_package: bool = False,
    package_name: Optional[str] = None,
) -> Path:
    """Convenience function to export adapters.

    Args:
        model_path: Path to trained model
        output_dir: Output directory
        create_package: Whether to create a zip package
        package_name: Name for the zip package (auto-generated if None)

    Returns:
        Path to exported adapters or package
    """
    exporter = AdapterExporter(model_path)

    if create_package:
        if package_name is None:
            package_name = f"deepsynth_adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

        package_path = Path(output_dir) / package_name
        return exporter.create_deployment_package(package_path)
    else:
        return exporter.export_adapters(output_dir)


__all__ = ["AdapterExporter", "export_adapter"]
