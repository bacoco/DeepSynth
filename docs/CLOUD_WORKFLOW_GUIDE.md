# 🌐 Guide de Workflow Cloud - Architecture Séparée

## Vue d'Ensemble

Ce guide décrit l'architecture séparée de DeepSynth:
- **Phase 1**: Génération de datasets (CPU, local/cloud) → Upload HuggingFace
- **Phase 2**: Fine-tuning (GPU, cloud) → Charge datasets depuis HuggingFace

### Avantages de cette Architecture

✅ **Séparation des Concerns**
- Génération d'images ≠ Fine-tuning
- CPU pour preprocessing, GPU pour training
- Équipes différentes peuvent travailler en parallèle

✅ **Réutilisabilité**
- Datasets générés une seule fois
- Réutilisables pour plusieurs expériences
- Partageables entre membres d'équipe

✅ **Parallélisation**
- Générer plusieurs datasets simultanément
- Machines CPU bon marché pour preprocessing
- GPU coûteux uniquement pour training

✅ **Efficacité Coût**
- CPU instances: $0.05/h pour génération
- GPU instances: $1-3/h uniquement quand nécessaire
- Storage HF gratuit (public) ou peu coûteux (privé)

---

## 📋 Architecture Complète

```
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: DATASET GENERATION (CPU)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Machine 1 (CPU):                                           │
│  ┌─────────────┐      ┌──────────────┐     ┌─────────────┐ │
│  │  MLSUM FR   │  →   │ Text-to-Image│  →  │ HuggingFace │ │
│  │   392k      │      │  Conversion  │     │   Dataset   │ │
│  └─────────────┘      └──────────────┘     └─────────────┘ │
│                                                             │
│  Machine 2 (CPU):                                           │
│  ┌─────────────┐      ┌──────────────┐     ┌─────────────┐ │
│  │  MLSUM ES   │  →   │ Text-to-Image│  →  │ HuggingFace │ │
│  │   266k      │      │  Conversion  │     │   Dataset   │ │
│  └─────────────┘      └──────────────┘     └─────────────┘ │
│                                                             │
│  Machine 3 (CPU):                                           │
│  ┌─────────────┐      ┌──────────────┐     ┌─────────────┐ │
│  │ CNN/DailyMail│  →  │ Text-to-Image│  →  │ HuggingFace │ │
│  │   287k      │      │  Conversion  │     │   Dataset   │ │
│  └─────────────┘      └──────────────┘     └─────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘

                              ⬇️

┌─────────────────────────────────────────────────────────────┐
│           PHASE 2: FINE-TUNING (GPU, HuggingFace)          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GPU Instance:                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │ HuggingFace  │  →  │ Combine      │  →  │ Fine-Tune   │ │
│  │ Datasets     │     │ Datasets     │     │ DeepSeek    │ │
│  │ (3 datasets) │     │              │     │ OCR         │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
│                                                             │
│                                             ⬇️              │
│                                      ┌─────────────┐        │
│                                      │ Trained     │        │
│                                      │ Model       │        │
│                                      └─────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Workflow Complet

### Étape 1: Génération des Datasets (Phase Preprocessing)

#### 1.1 Configuration Initiale

```bash
# Configurer les credentials HuggingFace
cp .env.example .env
# Éditer .env pour ajouter:
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
# HF_USERNAME=your-username
```

#### 1.2 Générer UN Dataset (Test)

```bash
# Générer MLSUM French avec limite (pour test)
python scripts/dataset_generation/generate_dataset_cloud.py \
    mlsum_fr \
    --max-samples 100 \
    --private  # ou sans --private pour public
```

**Sortie attendue**:
```
🚀 Génération du dataset: mlsum_fr
   Source: mlsum
   Target: deepsynth-mlsum-fr-images

📊 Processing split: train
   Loading mlsum (fr) - train
   Limited to 100 samples
   Processing 100 samples...
   ✅ Processed: 98/100 (errors: 2)

📊 Processing split: validation
   ...

⬆️  Uploading to HuggingFace: your-username/deepsynth-mlsum-fr-images

✅ Dataset uploadé: https://huggingface.co/datasets/your-username/deepsynth-mlsum-fr-images
```

#### 1.3 Générer TOUS les Datasets (Production)

**Option A: Séquentiellement sur une machine**
```bash
python scripts/dataset_generation/generate_dataset_cloud.py all
```

**Option B: En parallèle sur plusieurs machines**

Machine 1 (CPU):
```bash
python scripts/dataset_generation/generate_dataset_cloud.py mlsum_fr
python scripts/dataset_generation/generate_dataset_cloud.py mlsum_es
```

Machine 2 (CPU):
```bash
python scripts/dataset_generation/generate_dataset_cloud.py mlsum_de
python scripts/dataset_generation/generate_dataset_cloud.py cnn_dailymail
```

Machine 3 (CPU):
```bash
python scripts/dataset_generation/generate_dataset_cloud.py xsum
python scripts/dataset_generation/generate_dataset_cloud.py billsum
```

**Avantage**: 3x plus rapide car parallèle!

#### 1.4 Vérifier les Datasets Générés

```bash
python scripts/dataset_generation/generate_dataset_cloud.py mlsum_fr --list
```

**Sortie**:
```
📋 Datasets générés:
   ✅ your-username/deepsynth-mlsum-fr-images
   ✅ your-username/deepsynth-mlsum-es-images
   ✅ your-username/deepsynth-mlsum-de-images
   ✅ your-username/deepsynth-cnn-dailymail-images
   ✅ your-username/deepsynth-xsum-images
   ❌ your-username/deepsynth-billsum-images (pas encore généré)
```

---

### Étape 2: Fine-Tuning depuis le Cloud (Phase Training)

#### 2.1 Entraîner avec UN Dataset

```bash
python scripts/training/train_from_cloud_datasets.py \
    --datasets mlsum_fr \
    --batch-size 8 \
    --epochs 3 \
    --mixed-precision bf16 \
    --output-dir ./checkpoints/mlsum-fr-only
```

#### 2.2 Entraîner avec PLUSIEURS Datasets (Multilingue)

```bash
python scripts/training/train_from_cloud_datasets.py \
    --datasets mlsum_fr mlsum_es mlsum_de \
    --batch-size 4 \
    --epochs 3 \
    --mixed-precision bf16 \
    --output-dir ./checkpoints/multilingual
```

#### 2.3 Entraîner avec TOUS les Datasets

```bash
python scripts/training/train_from_cloud_datasets.py \
    --datasets all \
    --batch-size 4 \
    --epochs 3 \
    --output-dir ./checkpoints/all-datasets
```

#### 2.4 Configuration Avancée

```bash
python scripts/training/train_from_cloud_datasets.py \
    --datasets mlsum_fr cnn_dailymail \
    --max-train-samples 50000 \
    --max-val-samples 5000 \
    --batch-size 16 \
    --epochs 5 \
    --learning-rate 1e-5 \
    --mixed-precision fp16 \
    --num-workers 8 \
    --gradient-accumulation 2 \
    --output-dir ./checkpoints/custom
```

---

## 📊 Cas d'Usage Pratiques

### Cas 1: Recherche Rapide (Itération)

**Objectif**: Tester rapidement une idée

```bash
# Générer petit dataset (une fois)
python scripts/dataset_generation/generate_dataset_cloud.py \
    mlsum_fr --max-samples 1000

# Itérer rapidement sur différentes configs
python scripts/training/train_from_cloud_datasets.py \
    --datasets mlsum_fr \
    --max-train-samples 800 \
    --max-val-samples 200 \
    --epochs 1 \
    --batch-size 16
```

### Cas 2: Production Multilingue

**Objectif**: Modèle final multilingue

```bash
# Générer tous les datasets (une fois, peut prendre plusieurs jours)
# Paralléliser sur 6 machines CPU
for dataset in mlsum_fr mlsum_es mlsum_de cnn_dailymail xsum billsum
do
    python scripts/dataset_generation/generate_dataset_cloud.py $dataset &
done

# Attendre que tous soient uploadés sur HuggingFace

# Entraîner le modèle final (GPU puissant)
python scripts/training/train_from_cloud_datasets.py \
    --datasets all \
    --batch-size 32 \
    --epochs 10 \
    --learning-rate 2e-5 \
    --mixed-precision bf16 \
    --output-dir ./checkpoints/production
```

### Cas 3: Fine-Tuning Spécialisé

**Objectif**: Modèle spécialisé pour le domaine légal

```bash
# Générer uniquement BillSum
python scripts/dataset_generation/generate_dataset_cloud.py billsum

# Fine-tuning spécialisé
python scripts/training/train_from_cloud_datasets.py \
    --datasets billsum \
    --batch-size 8 \
    --epochs 5 \
    --output-dir ./checkpoints/legal-specialist
```

### Cas 4: Transfer Learning

**Objectif**: Pré-entraîner sur données générales, puis spécialiser

```bash
# Phase 1: Pré-entraînement sur données générales
python scripts/training/train_from_cloud_datasets.py \
    --datasets cnn_dailymail xsum \
    --epochs 3 \
    --output-dir ./checkpoints/general

# Phase 2: Fine-tuning sur domaine spécifique
python scripts/training/train_from_cloud_datasets.py \
    --datasets mlsum_fr \
    --model-name ./checkpoints/general/best_model \
    --epochs 2 \
    --learning-rate 1e-5 \
    --output-dir ./checkpoints/french-specialist
```

---

## 💰 Estimation de Coûts

### Génération de Datasets (Phase 1)

**Ressources**: CPU-only, ~4 cores, 16GB RAM

| Dataset | Samples | Temps Estimé | Coût AWS (c5.xlarge $0.17/h) |
|---------|---------|--------------|------------------------------|
| MLSUM FR | 392k | 20h | $3.40 |
| MLSUM ES | 266k | 14h | $2.38 |
| MLSUM DE | 220k | 12h | $2.04 |
| CNN/DM | 287k | 15h | $2.55 |
| XSum | 204k | 11h | $1.87 |
| BillSum | 22k | 2h | $0.34 |
| **TOTAL** | **1.29M** | **74h** | **$12.58** |

**Avec parallélisation (6 machines)**:
- Temps: 20h (le plus long)
- Coût: 6 × $0.17/h × 20h = **$20.40**

### Fine-Tuning (Phase 2)

**Ressources**: GPU (A100 ou V100), 40GB+ VRAM

| Configuration | GPU | Temps | Coût AWS (p3.2xlarge $3.06/h) |
|---------------|-----|-------|-------------------------------|
| Test (1k samples) | V100 | 0.5h | $1.53 |
| Medium (50k) | V100 | 5h | $15.30 |
| Full (1.29M) | A100 | 48h | $146.88 |

### Comparaison: Couplé vs Séparé

**Ancien (Couplé)**:
- GPU pour TOUT: génération + training
- Temps GPU: 74h + 48h = 122h
- Coût: 122h × $3.06/h = **$373.32**

**Nouveau (Séparé)**:
- CPU pour génération: 20h × $0.17/h × 6 = $20.40
- GPU pour training: 48h × $3.06/h = $146.88
- **Total: $167.28** (économie de **$206** soit 55%)

---

## 🔧 Configuration Avancée

### Datasets Privés vs Publics

**Privé** (par défaut recommandé):
```bash
python scripts/dataset_generation/generate_dataset_cloud.py \
    mlsum_fr --private
```

**Public** (pour partager avec la communauté):
```bash
python scripts/dataset_generation/generate_dataset_cloud.py \
    mlsum_fr  # sans --private
```

### Utiliser des Datasets d'un Autre Utilisateur

Si quelqu'un d'autre a déjà généré les datasets:

```python
# Modifier dans scripts/training/train_from_cloud_datasets.py
# Ligne ~30
AVAILABLE_DATASETS = {
    "mlsum_fr": "other-user/deepsynth-mlsum-fr-images",  # Changez le username
    ...
}
```

Ou créer votre propre mapping:
```bash
python scripts/training/train_from_cloud_datasets.py \
    --datasets mlsum_fr \
    --hf-dataset-prefix "other-user"
```

---

## 📈 Workflow CI/CD Recommandé

### GitHub Actions - Génération Automatique

```yaml
# .github/workflows/generate-datasets.yml
name: Generate Datasets

on:
  workflow_dispatch:
    inputs:
      dataset:
        description: 'Dataset to generate'
        required: true
        type: choice
        options:
          - mlsum_fr
          - mlsum_es
          - all

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Generate dataset
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_USERNAME: ${{ secrets.HF_USERNAME }}
        run: |
          python scripts/dataset_generation/generate_dataset_cloud.py \
            ${{ github.event.inputs.dataset }}
```

---

## ⚡ Conseils de Performance

### 1. Parallélisation Optimale

```bash
# Utiliser GNU Parallel pour lancer plusieurs générations
parallel -j 3 \
    python scripts/dataset_generation/generate_dataset_cloud.py {} \
    ::: mlsum_fr mlsum_es mlsum_de cnn_dailymail xsum billsum
```

### 2. Chunking pour Très Gros Datasets

Si un dataset est trop gros pour une seule machine:

```python
# Modifier le script pour générer par chunks
python scripts/dataset_generation/generate_dataset_cloud.py \
    mlsum_fr \
    --start-index 0 \
    --end-index 100000 \
    --chunk-id 1

python scripts/dataset_generation/generate_dataset_cloud.py \
    mlsum_fr \
    --start-index 100000 \
    --end-index 200000 \
    --chunk-id 2
```

### 3. Caching Local

Pour éviter de télécharger à chaque fois:

```bash
export HF_DATASETS_CACHE="/mnt/large-disk/hf-cache"
python scripts/training/train_from_cloud_datasets.py --datasets all
```

---

## 🎓 Bonnes Pratiques

### ✅ À FAIRE

1. **Générer une fois, utiliser plusieurs fois**
   - Générer les datasets et les uploader
   - Réutiliser pour différentes expériences

2. **Versioning des datasets**
   - Ajouter dates dans les noms: `deepsynth-mlsum-fr-images-2025-10`
   - Permet de comparer différentes versions

3. **Documentation des datasets**
   - Les READMEs sont auto-générés
   - Ajouter détails supplémentaires si nécessaire

4. **Tests avec petits échantillons**
   - Toujours tester avec `--max-samples 100` d'abord
   - Vérifier que tout fonctionne avant génération complète

### ❌ À ÉVITER

1. **Régénérer à chaque training**
   - Coûteux et inutile
   - Utiliser datasets cloud existants

2. **Datasets publics avec données sensibles**
   - Toujours utiliser `--private` pour données propriétaires

3. **Oublier la limite de samples**
   - Peut saturer le quota HuggingFace
   - Commencer petit puis scaler

---

## 🔍 Troubleshooting

### Problème: Dataset pas trouvé

**Erreur**: `Dataset not found: your-username/deepsynth-mlsum-fr-images`

**Solution**:
```bash
# Vérifier que le dataset existe
python scripts/dataset_generation/generate_dataset_cloud.py mlsum_fr --list

# Vérifier les credentials
echo $HF_TOKEN
echo $HF_USERNAME
```

### Problème: Out of Memory durant génération

**Solution**:
```bash
# Utiliser une machine avec plus de RAM
# Ou réduire le batch de traitement
# Modifier dans generate_dataset_cloud.py:
# Traiter par plus petits batches
```

### Problème: Upload HuggingFace échoue

**Solution**:
```bash
# Vérifier la connexion
huggingface-cli whoami

# Re-login si nécessaire
huggingface-cli login
```

---

## 📞 Support

- **Documentation**: Ce guide
- **Scripts**: `scripts/dataset_generation/` et `scripts/training/`
- **Issues**: GitHub Issues
- **HuggingFace**: https://huggingface.co/docs/datasets

---

**Prêt à scaler! 🚀**