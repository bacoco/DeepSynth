# PRD DeepSeek-OCR : Fine-Tuning pour Résumé Abstractif

## 📋 Vue d'ensemble du Projet

**Objectif** : Fine-tuner uniquement le décodeur DeepSeek-OCR (570M paramètres actifs) pour générer des résumés abstractifs à partir de documents compressés visuellement à 20x.

**Architecture Cible** :
```
Texte/PDF → Image 2D → DeepEncoder (frozen, 380M) → Tokens Visuels 20x → Décodeur MoE (fine-tuned, 570M) → Résumé
```

## 🔗 Références Officielles

### Code Source
- **Repo GitHub Principal** : https://github.com/deepseek-ai/DeepSeek-OCR
- **HuggingFace Model** : https://huggingface.co/deepseek-ai/DeepSeek-OCR
- **Paper ArXiv** : https://arxiv.org/abs/2510.18234
- **Code Modeling** : https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/main/modeling_deepseekocr.py

### Guides d'Installation
- **Guide Complet 2025** : https://skywork.ai/blog/ai-agent/how-to-install-run-deepseek-ocr-2025-guide/
- **vLLM Integration** : https://docs.vllm.ai/projects/recipes/en/latest/DeepSynth/DeepSeek-OCR.html
- **Simon Willison Demo** : https://simonwillison.net/2025/Oct/20/deepseek-ocr-claude-code/

## 📊 Datasets Hugging Face

### Résumé de Documents
| Dataset | URL HF | Champs | Taille | Usage |
|---------|--------|---------|---------|-------|
| CNN/DailyMail | `ccdv/cnn_dailymail` | `article`, `highlights` | 287k train | Principal |
| XSum | `EdinburghNLP/xsum` | `document`, `summary` | 204k train | Extrême |
| arXiv | `ccdv/arxiv-summarization` | `article`, `abstract` | Variable | Scientifique |
| Gigaword | `gigaword` | `document`, `summary` | 3.8M | Titres |
| FINDSum | `trigaten/findsum` | Multiple | 21k | Finance |

### Vision-Document
| Dataset | URL HF | Description | Usage |
|---------|--------|-------------|-------|
| Docmatix | `HuggingFaceM4/Docmatix` | 2.4M images, 9.5M Q/A | DocVQA |
| DocVQA | `docvqa` | 50k questions sur docs | OCR+QA |
| Document Haystack | `AmazonScience/document-haystack` | 400 variants, long docs | Multi-page |
| OmniDocBench | `opendatalab/OmniDocBench` | Benchmark diversifié | Évaluation |

## 🏗️ Architecture Technique

### Composants
1. **DeepEncoder** (380M params) : SAM + CLIP + Compression 16x
2. **Espace Latent** : Tokens visuels compressés (64-400 tokens selon mode)  
3. **Décodeur MoE** (570M params actifs sur 3B total) : Generation de texte

### Configuration Compression 20x
- **Précision OCR** : ~60% (acceptable pour résumé)
- **Ratio Information** : 1 token visuel ≈ 20 tokens texte
- **Avantage** : Perte d'info = abstraction naturelle

## ⚙️ Setup Environnement

### Prérequis
```bash
Python >= 3.9
CUDA >= 11.8
GPU: 16GB+ recommandé (RTX 4090, A100)
RAM: 32GB+ recommandé
```

### Installation
```bash
# Clone du repo officiel
git clone https://github.com/deepseek-ai/DeepSeek-OCR
cd DeepSeek-OCR

# Installation dépendances
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.46.0 tokenizers>=0.20.0
pip install datasets accelerate
pip install flash-attn --no-build-isolation
pip install -r requirements.txt

# Login Hugging Face pour datasets
huggingface-cli login
# Token depuis: https://huggingface.co/settings/tokens
```

## 💾 Préparation des Données

### Exemple CNN/DailyMail
```python
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import textwrap

# Chargement dataset
dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0")

def text_to_image(text, max_width=1800, max_height=2400):
    """Convertit texte en image PNG"""
    font = ImageFont.truetype("/usr/share/fonts/truetype/arial.ttf", 20)
    
    # Wrap text pour largeur
    lines = []
    for paragraph in text.split('\n'):
        wrapped = textwrap.fill(paragraph, width=80)
        lines.extend(wrapped.split('\n'))
    
    # Calcul dimensions
    line_height = 25
    img_height = min(len(lines) * line_height + 40, max_height)
    
    # Création image
    img = Image.new('RGB', (max_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    
    y = 20
    for line in lines[:90]:  # Limite pour éviter images trop grandes
        draw.text((20, y), line, font=font, fill='black')
        y += line_height
        
    return img

# Exemple usage
sample = dataset['train'][0]
img = text_to_image(sample['article'])
img.save('document.png')
target_summary = sample['highlights']
```

### Dataset Personnalisé
```python
import torch
from torch.utils.data import Dataset

class SummarizationDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_tokens=128):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Convertir article en image
        img = text_to_image(item['article'])
        img_path = f"temp_{idx}.png"
        img.save(img_path)
        
        # Tokenize target summary
        summary_tokens = self.tokenizer(
            item['highlights'],
            max_length=self.max_tokens,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'image_path': img_path,
            'labels': summary_tokens['input_ids'].squeeze(),
            'attention_mask': summary_tokens['attention_mask'].squeeze()
        }
```

## 🔧 Code Fine-Tuning

### Architecture Modifiée
```python
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
import torch

# Chargement modèle
model_name = "deepseek-ai/DeepSeek-OCR" 
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Freeze de l'encodeur
for name, param in model.named_parameters():
    if 'encoder' in name or 'vision' in name:
        param.requires_grad = False
        
# Vérification
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Paramètres entraînables: {trainable_params:,}")  # ~570M
```

### Configuration Entraînement
```python
training_args = TrainingArguments(
    output_dir="./deepsynth-summarizer",
    
    # Epochs & Batch
    num_train_epochs=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch = 32
    
    # Learning Rate
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    
    # Optimizations
    fp16=True,
    gradient_checkpointing=True,
    dataloader_pin_memory=True,
    
    # Logging & Saving
    logging_steps=25,
    save_steps=1000,
    eval_steps=500,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # Cleanup
    remove_unused_columns=False,
    report_to="tensorboard"
)
```

### Fonction d'Entraînement Custom
```python
def custom_forward(model, batch):
    """Forward pass personnalisé pour résumé"""
    
    # Encoder images avec DeepEncoder (frozen)
    with torch.no_grad():
        visual_tokens = model.encode_images(batch['images'])  # Shape: [B, seq, hidden]
    
    # Decoder pour résumé (trainable)
    decoder_outputs = model.decoder(
        inputs_embeds=visual_tokens,
        labels=batch['labels'],
        attention_mask=batch['attention_mask']
    )
    
    return decoder_outputs

# Wrapper pour Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = custom_forward(model, inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# Lancement entraînement
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## 📈 Évaluation

### Métriques ROUGE
```python
from datasets import load_metric
import numpy as np

rouge = load_metric('rouge')

def compute_rouge(predictions, references):
    """Calcul ROUGE-1, ROUGE-2, ROUGE-L"""
    results = rouge.compute(
        predictions=predictions,
        references=references
    )
    
    return {
        'rouge1': results['rouge1'].mid.fmeasure,
        'rouge2': results['rouge2'].mid.fmeasure, 
        'rougeL': results['rougeL'].mid.fmeasure,
    }

# Évaluation sur test set
def evaluate_model(model, test_dataloader):
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Génération
            outputs = model.generate(
                **batch,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                length_penalty=2.0
            )
            
            # Décodage
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            refs = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            
            predictions.extend(preds)
            references.extend(refs)
    
    return compute_rouge(predictions, references)
```

### Benchmarks Cibles
| Métrique | CNN/DM Target | XSum Target | Notes |
|----------|---------------|-------------|--------|
| ROUGE-1 | 40-45 | 35-40 | Overlap unigrammes |
| ROUGE-2 | 18-22 | 12-16 | Overlap bigrammes |  
| ROUGE-L | 37-42 | 32-37 | Plus longue séquence |

## 🚀 Inférence Production

### Script Génération
```python
def generate_summary(model, tokenizer, text, max_length=128):
    """Génère résumé depuis texte"""
    
    # Conversion texte -> image
    img = text_to_image(text)
    img.save("temp_input.png")
    
    # Prompt pour résumé
    prompt = "<image>\n<|grounding|>Summarize this document briefly."
    
    # Génération
    result = model.infer(
        tokenizer, 
        prompt=prompt,
        image_file="temp_input.png",
        max_new_tokens=max_length,
        temperature=0.7,
        do_sample=True
    )
    
    return result['result']

# Usage
summary = generate_summary(model, tokenizer, long_document)
print(f"Résumé: {summary}")
```

### Optimisations vLLM
```python
# Pour déploiement haute performance
from vllm import LLM, SamplingParams

llm = LLM(
    model="./deepsynth-summarizer",
    trust_remote_code=True,
    dtype="float16",
    gpu_memory_utilization=0.9,
    max_model_len=4096
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=128,
    stop=["<|endoftext|>"]
)

# Batch inference
summaries = llm.generate(prompts, sampling_params)
```

## 📋 Checklist Implémentation

### Phase 1: Setup
- [ ] Clone repo DeepSeek-OCR
- [ ] Installation environnement Python
- [ ] Login Hugging Face + téléchargement datasets
- [ ] Test modèle de base sur exemple

### Phase 2: Données  
- [ ] Implémentation text_to_image()
- [ ] Création SummarizationDataset
- [ ] Split train/val/test
- [ ] Vérification quality samples

### Phase 3: Fine-Tuning
- [ ] Freeze encodeur, unfreeze décodeur  
- [ ] Configuration TrainingArguments
- [ ] Lancement entraînement
- [ ] Monitoring loss/métriques

### Phase 4: Évaluation
- [ ] Calcul ROUGE sur test set
- [ ] Comparaison avec baselines
- [ ] Analyse qualitative outputs
- [ ] Tests edge cases

### Phase 5: Production
- [ ] Optimisation inférence vLLM
- [ ] Documentation API
- [ ] Tests performance GPU
- [ ] Déploiement containerisé

## 🔧 Debugging & Troubleshooting

### Problèmes Courants
1. **OOM (Out of Memory)** : Réduire batch_size, activer gradient_checkpointing
2. **Convergence lente** : Ajuster learning_rate, warmup_steps  
3. **Qualité résumés** : Tuner generation parameters (temperature, length_penalty)
4. **Images trop grandes** : Limiter résolution, text wrapping

### Ressources Debug
- **Issues GitHub** : https://github.com/deepseek-ai/DeepSeek-OCR/issues
- **Fine-tuning Issue** : https://github.com/deepseek-ai/DeepSeek-OCR/issues/43
- **Community Discussions** : https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions

## 📚 Références Complètes

### Papers & Research
- DeepSeek-OCR Paper: https://arxiv.org/abs/2510.18234
- LoRA vs Full Fine-tuning: https://arxiv.org/abs/2410.21228
- Compression Rate Summarization: https://arxiv.org/abs/2110.07936

### Tools & Frameworks
- Transformers: https://huggingface.co/docs/transformers/
- Datasets: https://huggingface.co/docs/datasets/
- vLLM: https://docs.vllm.ai/
- Unsloth: https://docs.unsloth.ai/

---

*Ce document contient toutes les informations nécessaires pour implémenter le fine-tuning DeepSeek-OCR pour résumé abstractif. Suivez les sections dans l'ordre pour une implémentation complète.*