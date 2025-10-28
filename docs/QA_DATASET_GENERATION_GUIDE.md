# Guide de Génération du Dataset Q&A

## 📋 Résumé Exécutif

Ce document décrit la génération d'un **dataset Q&A combiné unique** (`deepsynth-qa`) qui regroupe Natural Questions et MS MARCO en un seul dataset avec images pré-générées.

### Décisions Clés

✅ **UN SEUL DATASET COMBINÉ** plutôt que plusieurs datasets séparés
- Meilleur pour le training multi-domaine
- Batch mixing automatique
- Traçabilité préservée via `metadata.source`

✅ **GÉNÉRATION ONE-SHOT** directe (pas de datasets intermédiaires à merger)
- Traitement séquentiel Natural Questions → MS MARCO
- Upload incrémental dans un seul dataset final
- Interruption/reprise supportée

✅ **IMAGES PRÉ-GÉNÉRÉES** à résolution gundam (1600px)
- 0% overhead pendant le training
- Downscale possible vers tiny/base si besoin
- Extraction contextuelle intelligente

---

## 🎯 Composition du Dataset Final

### `deepsynth-qa` (~1.3M samples)

| Source | Samples | Caractéristiques | Qualité Moyenne |
|--------|---------|------------------|-----------------|
| **Natural Questions** | ~300k | Documents longs (avg 8,907 tokens) <br> Extraction contextuelle intelligente <br> Long answer priority | 70% excellent <br> 25% good |
| **MS MARCO** | ~1M | Documents courts (avg 200-500 tokens) <br> Pas d'extraction nécessaire <br> Short answers | 95%+ excellent |

**Total**: ~1.3M samples Q&A multilingues et multi-formats

---

## 📊 Taille Estimée du Dataset

### Natural Questions (300k samples)
```
Documents après extraction: ~1,500 tokens moyenne
Image gundam: 1600×1500px × 3 (RGB) ≈ 7.2 MB/sample
Total non-compressé: 300k × 7.2 MB ≈ 2.16 TB
Compressé PNG (5-10x): 216-432 GB
```

### MS MARCO (1M samples)
```
Documents courts: ~200 tokens moyenne
Image gundam: 1600×200px × 3 (RGB) ≈ 960 KB/sample
Total non-compressé: 1M × 960 KB ≈ 960 GB
Compressé PNG (5-10x): 96-192 GB
```

### 🎯 Total Estimé: **312-624 GB compressé sur HuggingFace**

C'est gérable! HuggingFace supporte des datasets bien plus larges.

---

## 🚀 Utilisation

### 1️⃣ Test Rapide (Recommandé pour débuter)

```bash
# Test avec 1000 samples par source (~2 minutes)
./generate_qa_dataset.sh --test

# Test avec nombre personnalisé
./generate_qa_dataset.sh --max-samples 5000
```

### 2️⃣ Génération Production Complète

```bash
# Génération complète (~6-12 heures)
./generate_qa_dataset.sh

# Ou directement avec Python
PYTHONPATH=./src python3 generate_qa_dataset.py
```

### 3️⃣ Options Avancées

```bash
# Seulement Natural Questions
python3 generate_qa_dataset.py --skip-marco

# Seulement MS MARCO
python3 generate_qa_dataset.py --skip-nq

# Custom samples pour chaque source
python3 generate_qa_dataset.py --nq-samples 50000 --marco-samples 100000

# Résolution différente
python3 generate_qa_dataset.py --resolution base

# Batch size personnalisé
python3 generate_qa_dataset.py --batch-size 10000
```

---

## 📦 Structure du Dataset

### Schéma des Samples

```python
{
    # Document et instruction
    "text": "DOCUMENT COMPLET (8,907 tokens pour NQ)",
    "instruction": "when is the last episode of walking dead season 8?",

    # Réponses
    "answer": "The eighth season premiered..." (LONG priority),
    "short_answer": "March 18, 2018",
    "long_answer": "The eighth season premiered...",

    # Positions (pour extraction future si besoin)
    "answer_start_token": 2114,
    "answer_end_token": 3538,

    # Image PRÉ-GÉNÉRÉE
    "image": <PIL.Image 1600×2224px>,

    # Qualité
    "quality": "excellent",
    "estimated_height": 2224,
    "token_count": 8907,  # Document COMPLET
    "extracted_token_count": 2224,  # Ce qui est dans l'image

    # Metadata avec ORIGINE
    "metadata": {
        "source": "natural_questions",  # ← TRAÇABILITÉ
        "original_index": 42,
        "answer_type": "long",
        "has_short": true,
        "has_long": true,
        "extraction_method": "contextual",
        "context_window": 800,
        "generation_resolution": "gundam",
        "quality_description": "Optimal readability (≤2200px)"
    }
}
```

### Filtrage par Source

```python
from datasets import load_dataset

# Charger dataset complet
dataset = load_dataset("baconnier/deepsynth-qa", split="train")

# Filtrer Natural Questions seulement
nq_dataset = dataset.filter(
    lambda x: x["metadata"]["source"] == "natural_questions"
)

# Filtrer MS MARCO seulement
marco_dataset = dataset.filter(
    lambda x: x["metadata"]["source"] == "ms_marco"
)

# Statistiques par source
from collections import Counter
sources = Counter(sample["metadata"]["source"] for sample in dataset)
print(sources)  # {'natural_questions': 300k, 'ms_marco': 1M}
```

---

## 🔧 Architecture d'Implémentation

### Flux de Génération One-Shot

```
┌─────────────────────────────────────────────────────────┐
│ GÉNÉRATION ONE-SHOT (generate_qa_dataset.py)           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1️⃣ Initialize IncrementalDatasetUploader              │
│     ├─ HuggingFace repo: deepsynth-qa                  │
│     ├─ Batch size: 5000                                │
│     └─ Duplicate tracking: (source, original_index)    │
│                                                          │
│  2️⃣ Process Natural Questions                          │
│     ├─ convert_natural_questions(streaming=True)       │
│     ├─ Intelligent contextual extraction               │
│     ├─ Pre-generate images (gundam 1600px)            │
│     ├─ Set metadata.source = "natural_questions"      │
│     └─ Upload batches incrementally (every 5k)        │
│                                                          │
│  3️⃣ Process MS MARCO                                   │
│     ├─ convert_ms_marco(streaming=True)                │
│     ├─ No extraction (docs already short)              │
│     ├─ Pre-generate images (gundam 1600px)            │
│     ├─ Set metadata.source = "ms_marco"               │
│     └─ Upload batches incrementally (every 5k)        │
│                                                          │
│  4️⃣ Result: Single combined dataset                    │
│     └─ deepsynth-qa with ~1.3M samples                │
└─────────────────────────────────────────────────────────┘
```

### Algorithme d'Extraction Intelligente

```python
# Pour chaque document Natural Questions:

1. Calculer token_count du document COMPLET

2. Décider stratégie:
   should_extract, window = should_extract_context(
       token_count,
       target_resolution="gundam"
   )

   # Seuil: max(2000, context_window * 2)
   # Pour gundam: max(2000, 800*2) = 2000

3. Si should_extract == False (doc ≤ 2000 tokens):
   → Utiliser document ENTIER pour l'image
   → extraction_method = "full_document"

4. Si should_extract == True (doc > 2000 tokens):
   → Extraire ±800 tokens autour de la réponse
   → extraction_method = "contextual"

5. Générer image à résolution gundam (1600px)

6. Stocker:
   - "text": Document COMPLET (flexibilité)
   - "image": Image PRÉ-GÉNÉRÉE
   - "token_count": Taille COMPLÈTE
   - "extracted_token_count": Ce qui est dans l'image
```

---

## 📈 Avantages de l'Approche Combinée

### ✅ Training Multi-Domaine

**Batch Mixing Automatique**:
```python
# Un batch contient naturellement un mix de sources
batch = [
    sample_nq_1,      # Natural Questions (long-form)
    sample_marco_1,   # MS MARCO (short)
    sample_nq_2,      # Natural Questions
    sample_marco_2,   # MS MARCO
    ...
]
# → Meilleure généralisation cross-domaine
```

### ✅ Gestion Simplifiée

```python
# UN SEUL dataset à charger
from datasets import load_dataset
dataset = load_dataset("baconnier/deepsynth-qa")

# VS plusieurs datasets séparés
nq_dataset = load_dataset("baconnier/deepsynth-nq")
marco_dataset = load_dataset("baconnier/deepsynth-marco")
# ... puis merge manuel
```

### ✅ Flexibilité Préservée

```python
# Filtrer par qualité
excellent_samples = dataset.filter(lambda x: x["quality"] == "excellent")

# Filtrer par source
nq_only = dataset.filter(lambda x: x["metadata"]["source"] == "natural_questions")

# Filtrer par type de réponse
long_answers = dataset.filter(lambda x: x["metadata"]["has_long"])

# Filtrer par extraction
contextual = dataset.filter(lambda x: x["metadata"]["extraction_method"] == "contextual")
```

---

## 🔍 Validation

### Test de l'Implémentation

```bash
# Valider l'implémentation
PYTHONPATH=./src python3 test_qa_implementation.py
```

**Résultats Attendus**:
```
✅ ALL TESTS PASSED!

Implementation validated:
  ✓ Images pre-generated at gundam resolution (1600px)
  ✓ Intelligent contextual extraction
  ✓ Full document stored in 'text' field
  ✓ Answer positions stored
  ✓ Dataset origin metadata properly set
  ✓ Quality indicators calculated
```

### Vérifier le Dataset Généré

```python
from datasets import load_dataset

# Charger dataset
dataset = load_dataset("baconnier/deepsynth-qa", split="train")

# Vérifier composition
from collections import Counter
sources = Counter(sample["metadata"]["source"] for sample in dataset)
print(f"Natural Questions: {sources['natural_questions']:,}")
print(f"MS MARCO: {sources['ms_marco']:,}")

# Vérifier images pré-générées
sample = dataset[0]
assert sample["image"] is not None, "Image should be pre-generated"
print(f"Image size: {sample['image'].size}")  # (1600, height)

# Vérifier metadata
assert "source" in sample["metadata"]
assert "generation_resolution" in sample["metadata"]
assert sample["metadata"]["generation_resolution"] == "gundam"
```

---

## 🎓 FAQ

### Q: Pourquoi un seul dataset plutôt que plusieurs?

**R**: Batch mixing automatique → meilleure généralisation. Le champ `metadata.source` permet de filtrer par source si nécessaire, donc vous gardez toute la flexibilité.

### Q: Combien de temps pour générer?

**R**:
- Test (1k samples/source): ~5-10 minutes
- Production complète (~1.3M): 6-12 heures

### Q: Quelle taille finale?

**R**: 312-624 GB compressé sur HuggingFace (gérable).

### Q: Puis-je interrompre et reprendre?

**R**: Oui! L'upload incrémental track les progress. Ctrl+C puis relancer le script.

### Q: Comment filtrer par source pendant le training?

**R**:
```python
# Option 1: Filtrer avant le training
nq_dataset = dataset.filter(lambda x: x["metadata"]["source"] == "natural_questions")

# Option 2: Stratified sampling
from torch.utils.data import WeightedRandomSampler
weights = [2.0 if x["metadata"]["source"] == "natural_questions" else 1.0
           for x in dataset]
sampler = WeightedRandomSampler(weights, len(dataset))
```

### Q: Les images sont-elles vraiment pré-générées?

**R**: OUI! Contrairement à l'ancienne approche (génération à la volée), les images sont maintenant générées UNE FOIS pendant la création du dataset et stockées sur HuggingFace. Training overhead = 0%.

### Q: Que contient le champ "text"?

**R**: Le document COMPLET (pas l'extrait). Pour Natural Questions, c'est le document original de 8,907 tokens en moyenne. L'extraction contextuelle est seulement pour l'image.

---

## 📚 Références

- [QA_FINAL_IMPLEMENTATION_PLAN.md](./QA_FINAL_IMPLEMENTATION_PLAN.md) - Plan détaillé d'implémentation
- [QA_QUALITY_INDICATORS_IMPLEMENTATION.md](./QA_QUALITY_INDICATORS_IMPLEMENTATION.md) - Indicateurs de qualité
- [QA_STREAMING_IMPLEMENTATION.md](./QA_STREAMING_IMPLEMENTATION.md) - Streaming dataset support
- [PRODUCTION_GUIDE.md](./PRODUCTION_GUIDE.md) - Guide de déploiement production

---

**Document Version**: 1.0
**Date**: 2025-10-28
**Auteur**: DeepSynth Team
**Status**: ✅ Implémentation complète et testée
