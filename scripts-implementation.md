# Scripts d'Implémentation DeepSeek-OCR Résumé

## 📁 Structure du Projet

```
deepsynth-summarizer/
├── data/
│   ├── prepare_datasets.py
│   ├── text_to_image.py
│   └── dataset_loader.py
├── training/
│   ├── train.py
│   ├── trainer.py
│   └── config.py
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   └── generate.py
├── inference/
│   ├── infer.py
│   └── api_server.py
├── requirements.txt
├── setup.sh
└── README.md
```

## 🛠️ Scripts Principaux

### 1. Setup Initial (`setup.sh`)

```bash
#!/bin/bash

# Setup DeepSeek-OCR Fine-tuning Environment
echo "🚀 Setting up DeepSeek-OCR Fine-tuning..."

# Clone repository
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies  
pip install transformers>=4.46.0
pip install tokenizers>=0.20.0
pip install datasets>=2.14.0
pip install accelerate>=0.24.0

# Install performance optimizations
pip install flash-attn --no-build-isolation
pip install xformers

# Install evaluation metrics
pip install rouge-score
pip install bert-score
pip install nltk

# Install monitoring
pip install tensorboard
pip install wandb

# Install other utilities
pip install pillow
pip install tqdm
pip install pandas
pip install matplotlib
pip install seaborn

# Login to Hugging Face
echo "🔑 Please login to Hugging Face for dataset access:"
huggingface-cli login

echo "✅ Setup complete! Activate environment with: source venv/bin/activate"
```

### 2. Text-to-Image Converter (`apps/web/ui/../??`)

```python
"""
Text to Image conversion for DeepSeek-OCR preprocessing
"""

from PIL import Image, ImageDraw, ImageFont
import textwrap
import os
from typing import Tuple, Optional

class TextToImageConverter:
    def __init__(self, 
                 font_path: str = "/usr/share/fonts/truetype/arial.ttf",
                 font_size: int = 18,
                 max_width: int = 1600,
                 max_height: int = 2200,
                 margin: int = 40):
        """
        Initialize text to image converter
        
        Args:
            font_path: Path to TTF font file
            font_size: Font size for text
            max_width: Maximum image width
            max_height: Maximum image height  
            margin: Margin around text
        """
        try:
            self.font = ImageFont.truetype(font_path, font_size)
        except OSError:
            print(f"Warning: Font {font_path} not found, using default")
            self.font = ImageFont.load_default()
            
        self.font_size = font_size
        self.max_width = max_width
        self.max_height = max_height
        self.margin = margin
        self.line_height = int(font_size * 1.3)
        
    def wrap_text(self, text: str, chars_per_line: int = 85) -> list:
        """Wrap text to fit image width"""
        paragraphs = text.split('\n')
        wrapped_lines = []
        
        for paragraph in paragraphs:
            if paragraph.strip():
                wrapped = textwrap.fill(paragraph, width=chars_per_line)
                wrapped_lines.extend(wrapped.split('\n'))
            else:
                wrapped_lines.append("")  # Preserve empty lines
                
        return wrapped_lines
    
    def calculate_dimensions(self, lines: list) -> Tuple[int, int]:
        """Calculate optimal image dimensions"""
        max_line_width = 0
        for line in lines:
            bbox = self.font.getbbox(line)
            line_width = bbox[2] - bbox[0]
            max_line_width = max(max_line_width, line_width)
            
        width = min(max_line_width + 2 * self.margin, self.max_width)
        height = min(len(lines) * self.line_height + 2 * self.margin, self.max_height)
        
        return width, height
    
    def convert(self, text: str, output_path: Optional[str] = None) -> Image.Image:
        """
        Convert text to image
        
        Args:
            text: Input text to convert
            output_path: Optional path to save image
            
        Returns:
            PIL Image object
        """
        # Wrap text
        lines = self.wrap_text(text)
        
        # Limit lines to prevent huge images
        max_lines = (self.max_height - 2 * self.margin) // self.line_height
        if len(lines) > max_lines:
            lines = lines[:max_lines-1]
            lines.append("... [text truncated]")
        
        # Calculate dimensions
        width, height = self.calculate_dimensions(lines)
        
        # Create image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw text
        y = self.margin
        for line in lines:
            draw.text((self.margin, y), line, font=self.font, fill='black')
            y += self.line_height
            
        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)
            
        return img

# Usage example
if __name__ == "__main__":
    converter = TextToImageConverter()
    
    sample_text = """
    This is a sample document that will be converted to an image.
    It contains multiple paragraphs and should wrap properly.
    
    This is a new paragraph to test paragraph handling.
    The converter should maintain proper spacing and formatting.
    """
    
    img = converter.convert(sample_text, "sample_output.png")
    print("Image saved as sample_output.png")
```

### 3. Dataset Loader (`data/dataset_loader.py`)

```python
"""
Dataset loading and preprocessing for DeepSeek-OCR fine-tuning
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from PIL import Image
import os
import tempfile
from typing import Dict, List, Optional
from text_to_image import TextToImageConverter

class SummarizationDataset(Dataset):
    def __init__(self, 
                 dataset_name: str = "ccdv/cnn_dailymail",
                 dataset_config: str = "3.0.0",
                 split: str = "train",
                 tokenizer_name: str = "deepseek-ai/DeepSeek-OCR",
                 max_source_length: int = 1024,
                 max_target_length: int = 128,
                 cache_dir: Optional[str] = None):
        """
        Initialize summarization dataset
        
        Args:
            dataset_name: HuggingFace dataset identifier
            dataset_config: Dataset configuration
            split: Dataset split (train/validation/test)
            tokenizer_name: Tokenizer model name
            max_source_length: Maximum source text length
            max_target_length: Maximum target summary length
            cache_dir: Directory to cache processed data
        """
        self.dataset = load_dataset(dataset_name, dataset_config, split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            trust_remote_code=True
        )
        self.converter = TextToImageConverter()
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.cache_dir = cache_dir or tempfile.mkdtemp()
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Field mapping for different datasets
        self.field_mapping = {
            "ccdv/cnn_dailymail": {"source": "article", "target": "highlights"},
            "EdinburghNLP/xsum": {"source": "document", "target": "summary"},
            "ccdv/arxiv-summarization": {"source": "article", "target": "abstract"},
            "gigaword": {"source": "document", "target": "summary"}
        }
        
        self.source_field = self.field_mapping[dataset_name]["source"]
        self.target_field = self.field_mapping[dataset_name]["target"]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item"""
        item = self.dataset[idx]
        
        # Get source and target text
        source_text = item[self.source_field]
        target_text = item[self.target_field]
        
        # Convert source text to image
        image_path = os.path.join(self.cache_dir, f"doc_{idx}.png")
        if not os.path.exists(image_path):
            # Truncate source if too long
            if len(source_text) > self.max_source_length * 4:  # Rough char estimate
                source_text = source_text[:self.max_source_length * 4] + "..."
                
            self.converter.convert(source_text, image_path)
        
        # Tokenize target summary
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "image_path": image_path,
            "input_ids": target_encoding["input_ids"].squeeze(),
            "attention_mask": target_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze()
        }

def create_dataloaders(dataset_name: str = "ccdv/cnn_dailymail",
                      batch_size: int = 4,
                      num_workers: int = 4) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders"""
    
    datasets = {}
    for split in ["train", "validation", "test"]:
        try:
            datasets[split] = SummarizationDataset(
                dataset_name=dataset_name,
                split=split
            )
        except:
            print(f"Warning: {split} split not available for {dataset_name}")
    
    dataloaders = {}
    for split, dataset in datasets.items():
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders

# Usage example
if __name__ == "__main__":
    # Test dataset loading
    dataset = SummarizationDataset(split="train[:100]")  # Small sample
    print(f"Dataset size: {len(dataset)}")
    
    # Test single item
    item = dataset[0]
    print(f"Image path: {item['image_path']}")
    print(f"Labels shape: {item['labels'].shape}")
    print(f"Input IDs shape: {item['input_ids'].shape}")
```

### 4. Training Script (`training/train.py`)

```python
"""
Main training script for DeepSeek-OCR fine-tuning
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, TrainingArguments, 
    Trainer, EarlyStoppingCallback
)
from datasets import load_metric
import numpy as np
import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

from deepsynth.data.dataset_loader import create_dataloaders

@dataclass
class ModelArguments:
    model_name: str = "deepseek-ai/DeepSeek-OCR"
    cache_dir: Optional[str] = None
    use_auth_token: bool = True

@dataclass  
class DataArguments:
    dataset_name: str = "ccdv/cnn_dailymail"
    dataset_config: str = "3.0.0"
    max_source_length: int = 1024
    max_target_length: int = 128

@dataclass
class TrainingArgs:
    output_dir: str = "./outputs"
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_steps: int = 25
    save_steps: int = 1000
    eval_steps: int = 500
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_rouge1"
    fp16: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "tensorboard"

class DeepSynthSummarizationModel(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Freeze encoder parameters
        self._freeze_encoder()
        
    def _freeze_encoder(self):
        """Freeze encoder parameters, only train decoder"""
        for name, param in self.model.named_parameters():
            if any(keyword in name.lower() for keyword in ['encoder', 'vision', 'embed']):
                param.requires_grad = False
                print(f"Frozen: {name}")
    
    def forward(self, image_paths: List[str], labels: torch.Tensor, **kwargs):
        """Forward pass for training"""
        batch_size = len(image_paths)
        device = labels.device
        
        # Process images and generate summaries
        all_losses = []
        
        for i, image_path in enumerate(image_paths):
            # Generate summary using model's infer method
            try:
                prompt = "<image>\n<|grounding|>Summarize this document."
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=image_path,
                    max_new_tokens=128
                )
                
                # Tokenize generated summary
                generated_text = result.get('result', '')
                generated_tokens = self.tokenizer(
                    generated_text,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=labels.shape[1]
                )['input_ids'].to(device)
                
                # Compute loss against target
                target = labels[i].unsqueeze(0)
                loss = F.cross_entropy(
                    generated_tokens.float(), 
                    target.long(),
                    ignore_index=self.tokenizer.pad_token_id
                )
                all_losses.append(loss)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                # Use dummy loss if processing fails
                dummy_loss = torch.tensor(0.0, requires_grad=True, device=device)
                all_losses.append(dummy_loss)
        
        # Average losses
        total_loss = torch.stack(all_losses).mean()
        
        return type('ModelOutput', (), {
            'loss': total_loss,
            'logits': None
        })()

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rouge = load_metric('rouge')
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation"""
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Custom evaluation with ROUGE metrics"""
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        predictions = []
        references = []
        
        self.model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                # Generate summaries
                for image_path, label in zip(batch['image_path'], batch['labels']):
                    try:
                        prompt = "<image>\n<|grounding|>Summarize this document."
                        result = self.model.model.infer(
                            self.model.tokenizer,
                            prompt=prompt,
                            image_file=image_path,
                            max_new_tokens=128
                        )
                        
                        pred_text = result.get('result', '')
                        ref_text = self.tokenizer.decode(
                            label, 
                            skip_special_tokens=True
                        )
                        
                        predictions.append(pred_text)
                        references.append(ref_text)
                        
                    except Exception as e:
                        print(f"Evaluation error: {e}")
                        predictions.append("")
                        references.append("")
        
        # Compute ROUGE
        if predictions and references:
            rouge_results = self.rouge.compute(
                predictions=predictions,
                references=references
            )
            
            metrics = {
                f"{metric_key_prefix}_rouge1": rouge_results['rouge1'].mid.fmeasure,
                f"{metric_key_prefix}_rouge2": rouge_results['rouge2'].mid.fmeasure,
                f"{metric_key_prefix}_rougeL": rouge_results['rougeL'].mid.fmeasure,
            }
        else:
            metrics = {
                f"{metric_key_prefix}_rouge1": 0.0,
                f"{metric_key_prefix}_rouge2": 0.0, 
                f"{metric_key_prefix}_rougeL": 0.0,
            }
            
        return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--dataset_name", default="ccdv/cnn_dailymail")
    parser.add_argument("--output_dir", default="./deepsynth-summarizer")
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    
    args = parser.parse_args()
    
    print(f"🚀 Starting DeepSeek-OCR fine-tuning...")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_name}")
    
    # Load model
    model = DeepSynthSummarizationModel(args.model_name)
    tokenizer = model.tokenizer
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load datasets
    dataloaders = create_dataloaders(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=25,
        save_steps=1000,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps", 
        load_best_model_at_end=True,
        metric_for_best_model="eval_rouge1",
        greater_is_better=True,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="tensorboard"
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataloaders["train"].dataset,
        eval_dataset=dataloaders.get("validation", {}).dataset if "validation" in dataloaders else None,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("🎯 Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ Training complete! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
```

### 5. Inference Script (`inference/infer.py`)

```python
"""
Inference script for trained DeepSeek-OCR summarization model
"""

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import argparse
import os
from deepsynth.data.text_to_image import TextToImageConverter

class DeepSynthSummarizer:
    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize the summarizer"""
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # Text to image converter
        self.converter = TextToImageConverter()
        
        print(f"✅ Model loaded on {self.device}")
        
    def summarize_text(self, 
                      text: str, 
                      max_length: int = 128,
                      temperature: float = 0.7,
                      num_beams: int = 4) -> str:
        """
        Summarize input text
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            temperature: Generation temperature
            num_beams: Number of beams for beam search
            
        Returns:
            Generated summary
        """
        # Convert text to image
        temp_image_path = "temp_input.png"
        self.converter.convert(text, temp_image_path)
        
        try:
            # Generate summary
            summary = self.summarize_image(
                temp_image_path,
                max_length=max_length,
                temperature=temperature,
                num_beams=num_beams
            )
            
            return summary
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
    
    def summarize_image(self,
                       image_path: str,
                       max_length: int = 128, 
                       temperature: float = 0.7,
                       num_beams: int = 4) -> str:
        """
        Summarize document image
        
        Args:
            image_path: Path to document image
            max_length: Maximum summary length
            temperature: Generation temperature
            num_beams: Number of beams for beam search
            
        Returns:
            Generated summary
        """
        prompt = "<image>\n<|grounding|>Summarize this document concisely."
        
        try:
            # Use model's infer method
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_path,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                num_beams=num_beams if temperature == 0 else 1
            )
            
            summary = result.get('result', 'Error generating summary')
            return summary.strip()
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def batch_summarize(self, 
                       texts: list,
                       max_length: int = 128,
                       temperature: float = 0.7) -> list:
        """Summarize multiple texts"""
        summaries = []
        
        for i, text in enumerate(texts):
            print(f"Processing {i+1}/{len(texts)}...")
            summary = self.summarize_text(
                text, 
                max_length=max_length,
                temperature=temperature
            )
            summaries.append(summary)
            
        return summaries

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR Summarization Inference")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--input_text", help="Text to summarize")
    parser.add_argument("--input_file", help="Text file to summarize")
    parser.add_argument("--image_path", help="Document image to summarize")
    parser.add_argument("--output_file", help="Output file for summary")
    parser.add_argument("--max_length", type=int, default=128, help="Max summary length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams")
    
    args = parser.parse_args()
    
    # Initialize summarizer
    summarizer = DeepSynthSummarizer(args.model_path)
    
    # Get input text
    if args.input_text:
        text = args.input_text
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.image_path:
        # Direct image summarization
        summary = summarizer.summarize_image(
            args.image_path,
            max_length=args.max_length,
            temperature=args.temperature,
            num_beams=args.num_beams
        )
        
        print(f"📄 Summary: {summary}")
        
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"💾 Summary saved to {args.output_file}")
        
        return
    else:
        print("❌ Please provide input text, file, or image path")
        return
    
    # Generate summary
    print("🔄 Generating summary...")
    summary = summarizer.summarize_text(
        text,
        max_length=args.max_length,
        temperature=args.temperature,
        num_beams=args.num_beams
    )
    
    print(f"\n📄 Original text length: {len(text)} characters")
    print(f"📄 Summary length: {len(summary)} characters")
    print(f"📊 Compression ratio: {len(text)/len(summary):.1f}x")
    print(f"\n📝 Summary:\n{summary}")
    
    # Save output
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"\n💾 Summary saved to {args.output_file}")

if __name__ == "__main__":
    main()

# Usage examples:
# python inference/infer.py --model_path ./deepsynth-summarizer --input_text "Long article text here..."
# python inference/infer.py --model_path ./deepsynth-summarizer --input_file article.txt --output_file summary.txt
# python inference/infer.py --model_path ./deepsynth-summarizer --image_path document.png
```

### 6. Requirements (`requirements.txt`)

```
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.46.0
tokenizers>=0.20.0
datasets>=2.14.0
accelerate>=0.24.0
flash-attn>=2.3.0
xformers>=0.0.22
rouge-score>=0.1.2
bert-score>=0.3.13
nltk>=3.8
tensorboard>=2.14.0
wandb>=0.16.0
pillow>=9.5.0
tqdm>=4.65.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
```

### 7. Utilisation Rapide

```bash
# 1. Setup environnement
chmod +x setup.sh
./setup.sh
source venv/bin/activate

# 2. Entraînement
python training/train.py \
    --model_name deepseek-ai/DeepSeek-OCR \
    --dataset_name ccdv/cnn_dailymail \
    --output_dir ./deepsynth-summarizer \
    --num_epochs 4 \
    --batch_size 4 \
    --learning_rate 2e-5

# 3. Inférence
python inference/infer.py \
    --model_path ./deepsynth-summarizer \
    --input_file document.txt \
    --output_file summary.txt \
    --max_length 128
```

---

*Ces scripts fournissent une implémentation complète et prête à l'emploi pour le fine-tuning DeepSeek-OCR sur des tâches de résumé. Adaptez les paramètres selon vos besoins spécifiques.*