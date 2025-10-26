# Installation Guide - DeepSynth

## Quick Start (Mac Studio ou tout autre Mac)

```bash
# 1. Clone le repo
git clone https://github.com/bacoco/DeepSynth.git
cd DeepSynth

# 2. Configure ton token HuggingFace
cp .env.example .env
nano .env  # Ajouter HF_TOKEN=ton_token_ici

# 3. Lance le script (il installe TOUT automatiquement)
./generate_all_datasets.sh
```

**C'est TOUT!** Le script détecte automatiquement macOS et installe les bonnes dépendances.

## Mise à Jour

```bash
cd DeepSynth
./update.sh
```

## Ce Qui Se Passe Automatiquement

### Sur macOS (comme ton Mac Studio)
✅ Installe Python venv
✅ Installe PyTorch CPU
✅ Installe **requirements-base.txt** (SANS xformers qui ne compile pas sur Mac)
✅ Télécharge et installe DejaVu Sans (fonts Unicode pour accents)
✅ Lance la génération des 7 datasets en parallèle

**Note**: xformers n'est PAS nécessaire pour la génération de datasets. Il n'est utile QUE pour l'entraînement GPU sur Linux.

### Sur Linux avec GPU
✅ Installe PyTorch CUDA
✅ Installe **requirements-base.txt** + **requirements-training.txt** (avec xformers)
✅ Installe fonts via apt/yum
✅ Prêt pour l'entraînement GPU

## Structure des Requirements

```
requirements-base.txt       → Pour génération de datasets (macOS, Linux CPU)
requirements-training.txt   → GPU training seulement (Linux + CUDA)
requirements.txt            → TOUT (peut échouer sur macOS à cause de xformers)
```

## Résolution de Problèmes

### Erreur "xformers failed to build"
✅ **Normal sur macOS** - Le script utilise automatiquement `requirements-base.txt` qui ne contient pas xformers.

Si vous avez lancé `pip install -r requirements.txt` manuellement, faites:
```bash
pip uninstall xformers
pip install -r requirements-base.txt
```

### "Font not found" ou accents mal affichés
Le script installe automatiquement DejaVu Sans. Si problème:
```bash
# macOS
sudo cp /tmp/dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf /Library/Fonts/

# Linux
sudo apt-get install fonts-dejavu
```

### Vérifier l'Installation

```bash
source venv/bin/activate
python -c "
from deepsynth.pipelines import ParallelDatasetsPipeline
pipeline = ParallelDatasetsPipeline(max_workers=2)
print(f'✅ {len(pipeline.datasets_config)} datasets configurés')
"
```

## Workflow Complet

### Mac Studio (Génération de Datasets)
```bash
git clone https://github.com/bacoco/DeepSynth.git
cd DeepSynth
cp .env.example .env
nano .env  # HF_TOKEN=...
./generate_all_datasets.sh
# ⏱️  6-12 heures → 7 datasets sur HuggingFace
```

### Linux GPU (Entraînement)
```bash
# Utiliser les datasets créés sur Mac
git clone https://github.com/bacoco/DeepSynth.git
cd DeepSynth
./setup.sh  # Installe CUDA, xformers, etc.
source venv/bin/activate

# Entraîner avec un dataset
python -m deepsynth.training.train \
    --use-deepseek-ocr \
    --hf-dataset baconnier/deepsynth-fr \
    --output ./model-fr
```

## FAQ

**Q: Dois-je installer xformers sur Mac?**
R: Non! xformers ne compile pas sur macOS et n'est pas nécessaire pour générer les datasets.

**Q: Puis-je entraîner un modèle sur Mac?**
R: Possible mais très lent (CPU). Recommandé: générer datasets sur Mac, entraîner sur Linux GPU.

**Q: Comment mettre à jour après un git pull?**
R: `./update.sh` - Il détecte l'OS et installe les bonnes dépendances.

**Q: Quel Python version?**
R: Python 3.9+ requis. Le script vérifie automatiquement.

---

**Support**: Consultez `QUICKSTART.md` ou `CLAUDE.md` pour plus de détails.
