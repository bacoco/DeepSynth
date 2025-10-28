# Plan d'Implémentation Final - Q&A Datasets avec Images Pré-générées

## 📋 Contexte et Décisions

### Problème Initial
- Documents Natural Questions très longs (moyenne 8,907 tokens)
- Extraction contextuelle faite à la génération du dataset
- Qualité calculée sans connaître la résolution de training
- Décisions prises au mauvais moment (génération vs training)

### Décisions Finales

**DÉCISION 1: Stocker document COMPLET dans le dataset**
- Raison: Flexibilité maximale, pas de perte d'information
- Le texte complet permet extraction contextuelle adaptative au training

**DÉCISION 2: PRÉ-GÉNÉRER les images à résolution MAXIMALE (gundam = 1600px)**
- Raison: Stockage HuggingFace géré, pas de contrainte d'espace
- Permet downscale vers tiny/base sans perte de qualité
- Training 0% overhead (pas de génération à la volée)
- Meilleure qualité possible pour tous les cas d'usage

**DÉCISION 3: Extraction contextuelle INTELLIGENTE à la génération**
- Basée sur résolution cible (gundam = 1600px)
- Si document > capacité résolution → extraction contextuelle
- Si document ≤ capacité résolution → document complet
- Seuil dynamique: `max(2000, context_window * 2)`

**DÉCISION 4: Priorité LONG answer**
- Long answer contient 15-142x plus de contexte
- Meilleur pour training du modèle
- Fallback sur short answer si pas de long

**DÉCISION 5: Supprimer FiQA**
- Dataset sans vraies réponses
- Inutilisable pour training

---

## 🎯 Architecture Finale

### Flux de Données

```
┌─────────────────────────────────────────────────────────────────┐
│ GÉNÉRATION DATASET (une fois)                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Natural Questions Sample                                        │
│  ├─ Document: 8,907 tokens (complet)                           │
│  ├─ Question: "when is the last episode..."                    │
│  ├─ Short answer: "March 18, 2018"                             │
│  ├─ Long answer: "The eighth season premiered..." (1,424 tok)  │
│  └─ Answer positions: start=2114, end=3538                     │
│                                                                  │
│  ▼ DÉCISION: Doc trop long pour gundam (1600px)                │
│                                                                  │
│  Extraction Contextuelle                                        │
│  ├─ Context window: 800 tokens (gundam optimal)                │
│  ├─ Extraction: tokens[2114-800 : 3538+800]                   │
│  └─ Résultat: ~2,224 tokens (contexte pertinent)              │
│                                                                  │
│  ▼ GÉNÉRATION IMAGE                                             │
│                                                                  │
│  TextToImageConverter                                           │
│  ├─ Résolution: gundam (1600×1600)                             │
│  ├─ Input: 2,224 tokens                                        │
│  └─ Output: Image 1600×2224px (excellent quality)             │
│                                                                  │
│  ▼ STOCKAGE HUGGINGFACE                                         │
│                                                                  │
│  {                                                              │
│    "text": "DOCUMENT COMPLET 8,907 tokens",                    │
│    "instruction": "when is the last episode...",               │
│    "answer": "The eighth season premiered..." (LONG priority)  │
│    "short_answer": "March 18, 2018",                           │
│    "long_answer": "The eighth season premiered...",            │
│    "answer_start_token": 2114,                                 │
│    "answer_end_token": 3538,                                   │
│    "image": <PIL.Image 1600×2224>,  ← PRÉ-GÉNÉRÉE             │
│    "quality": "excellent",                                      │
│    "estimated_height": 2224,                                   │
│    "token_count": 8907,                                        │
│    "extracted_token_count": 2224,                              │
│    "metadata": {                                               │
│      "extraction_method": "contextual",                        │
│      "context_window": 800,                                    │
│      "generation_resolution": "gundam",                        │
│      ...                                                       │
│    }                                                           │
│  }                                                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ TRAINING (répété)                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  InstructionDataset.__getitem__(idx)                            │
│  ├─ Charger sample depuis HuggingFace                          │
│  ├─ Récupérer image PRÉ-GÉNÉRÉE (1600×2224)                   │
│  ├─ Appliquer transform si nécessaire (resize/augmentation)    │
│  │   - Training en "tiny" → resize 1600×2224 → 512×512        │
│  │   - Training en "base" → resize 1600×2224 → 1024×1024      │
│  │   - Training en "gundam" → utiliser tel quel (optimal!)    │
│  └─ Return sample                                              │
│                                                                  │
│  ⏱️  Temps: 0ms génération (image déjà prête!)                  │
│  📊 Overhead: 0%                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Changements Détaillés par Fichier

### 1. `src/deepsynth/data/quality_calculator.py`

**Ajouter fonction pour calcul context window optimal**

```python
def calculate_optimal_context_window(
    target_resolution: str = "gundam",
    chars_per_line: int = 100,
    line_height_px: int = 20,
) -> int:
    """
    Calculate optimal context window for target resolution.

    Args:
        target_resolution: "tiny", "small", "base", "large", "gundam"
        chars_per_line: Characters per line (default: 100)
        line_height_px: Line height in pixels (default: 20)

    Returns:
        Optimal context window in tokens

    Example:
        >>> calculate_optimal_context_window("gundam")  # 1600px height
        800  # tokens (1600px / 20px/line * 100chars/line / 5chars/token)

        >>> calculate_optimal_context_window("base")  # 1024px height
        512  # tokens
    """
    from .transforms.text_to_image import DEEPSEEK_OCR_RESOLUTIONS

    if target_resolution not in DEEPSEEK_OCR_RESOLUTIONS:
        raise ValueError(f"Unknown resolution: {target_resolution}")

    target_height = DEEPSEEK_OCR_RESOLUTIONS[target_resolution][1]  # (width, height)

    # Calculate tokens that fit in target height
    lines_per_image = target_height / line_height_px
    chars_per_image = lines_per_image * chars_per_line
    tokens_per_image = chars_per_image / CHARS_PER_TOKEN

    # Context window is half of capacity (for before + after)
    context_window = int(tokens_per_image / 2)

    return context_window
```

**Ajouter fonction pour décision extraction**

```python
def should_extract_context(
    token_count: int,
    target_resolution: str = "gundam",
    min_threshold: int = 2000,
) -> tuple[bool, int]:
    """
    Decide if contextual extraction is needed based on document size and resolution.

    Args:
        token_count: Document token count
        target_resolution: Target resolution for image generation
        min_threshold: Minimum threshold before considering extraction

    Returns:
        Tuple of (should_extract, context_window_if_needed)

    Example:
        >>> should_extract_context(500, "gundam")
        (False, 0)  # Doc court, pas besoin d'extraction

        >>> should_extract_context(10000, "gundam")
        (True, 800)  # Doc long, extraction avec window=800
    """
    context_window = calculate_optimal_context_window(target_resolution)

    # Seuil dynamique: max(min_threshold, context_window * 2)
    extraction_threshold = max(min_threshold, context_window * 2)

    if token_count <= extraction_threshold:
        # Document assez court, pas besoin d'extraction
        return (False, 0)
    else:
        # Document trop long, extraction nécessaire
        return (True, context_window)
```

---

### 2. `src/deepsynth/data/dataset_converters/natural_questions.py`

**CHANGEMENTS MAJEURS**

**A. Modifier signature de `convert_natural_questions()`**

```python
def convert_natural_questions(
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = True,
    target_resolution: str = "gundam",  # ← NOUVEAU: Résolution pour pré-génération
) -> InstructionDataset:
    """
    Convert Natural Questions to instruction format with pre-generated images.

    Args:
        split: Dataset split ("train", "validation")
        max_samples: Maximum number of samples (None = all)
        streaming: Use streaming mode
        target_resolution: Target resolution for pre-generation ("gundam" recommended)

    Returns:
        InstructionDataset with pre-generated images at target resolution
    """
```

**B. Calculer context window optimal**

```python
from ..quality_calculator import should_extract_context, calculate_optimal_context_window

# Au début de la fonction
context_window = calculate_optimal_context_window(target_resolution)
LOGGER.info(f"Target resolution: {target_resolution}")
LOGGER.info(f"Optimal context window: {context_window} tokens")
```

**C. Décision extraction intelligente**

```python
# Après avoir extrait short et long answers...

# Calculer longueur document complet
full_document_token_count = len(text_tokens)

# Décider stratégie d'extraction
should_extract, extraction_window = should_extract_context(
    full_document_token_count,
    target_resolution
)

if should_extract:
    # Document trop long → extraction contextuelle
    if answer_start is not None and answer_end is not None:
        document_text = extract_contextual_window(
            text_tokens,
            answer_start,
            answer_end,
            context_before=extraction_window,
            context_after=extraction_window,
        )
        extraction_method = "contextual"
        extracted_token_count = len(document_text.split())
    else:
        # Cas rare: pas de positions, utiliser answer text
        document_text = primary_answer
        extraction_method = "answer_only"
        extracted_token_count = len(document_text.split())
else:
    # Document assez court → garder ENTIER
    document_text = " ".join(str(t) for t in text_tokens)
    extraction_method = "full_document"
    extracted_token_count = full_document_token_count
```

**D. Générer image à résolution cible**

```python
from ..transforms.text_to_image import TextToImageConverter

# Générer image à résolution cible
converter = TextToImageConverter(max_width=1600, max_height=10000)  # Gundam width
image = converter.convert(document_text)

# Note: L'image sera automatiquement redimensionnée si trop grande
# mais avec max_height=10000, on peut gérer même les docs très longs
```

**E. Calculer quality indicators**

```python
# Calculer qualité basée sur document EXTRAIT (pas complet)
quality, quality_desc, estimated_height = calculate_quality(extracted_token_count)
```

**F. Stocker TOUTES les informations**

```python
converted_samples.append({
    # Texte et instruction
    "text": " ".join(str(t) for t in text_tokens),  # ← DOCUMENT COMPLET
    "instruction": question.strip(),

    # Answers (long priority)
    "answer": primary_answer.strip(),
    "short_answer": short_answer_text.strip(),
    "long_answer": long_answer_text.strip(),

    # Positions pour extraction future si besoin
    "answer_start_token": answer_start,  # ← NOUVEAU
    "answer_end_token": answer_end,      # ← NOUVEAU

    # Image pré-générée
    "image": image,  # ← IMAGE PRÉ-GÉNÉRÉE À RÉSOLUTION CIBLE

    # Quality indicators
    "quality": quality,
    "estimated_height": estimated_height,
    "token_count": full_document_token_count,  # ← Document COMPLET
    "extracted_token_count": extracted_token_count,  # ← Document EXTRAIT (pour image)

    # Metadata étendu
    "metadata": {
        "source": "natural_questions",
        "original_index": idx,
        "answer_type": answer_type,
        "has_short": bool(short_answer_text),
        "has_long": bool(long_answer_text),
        "extraction_method": extraction_method,  # "contextual", "full_document", "answer_only"
        "context_window": extraction_window if should_extract else 0,
        "generation_resolution": target_resolution,  # ← NOUVEAU
        "quality_description": quality_desc,
    },
})
```

---

### 3. `src/deepsynth/data/dataset_converters/ms_marco.py`

**Changements similaires mais plus simples** (MS MARCO a déjà des docs courts)

```python
def convert_ms_marco(
    config: str = "v2.1",
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = True,
    target_resolution: str = "gundam",  # ← NOUVEAU
) -> InstructionDataset:
```

**Génération d'image** (MS MARCO raremen besoin d'extraction):

```python
from ..transforms.text_to_image import TextToImageConverter

# MS MARCO a généralement des passages courts, pas besoin d'extraction
converter = TextToImageConverter(max_width=1600, max_height=10000)
image = converter.convert(document)

# Calculer quality
token_count = len(document.split())
quality, quality_desc, estimated_height = calculate_quality(token_count)

converted_samples.append({
    "text": document.strip(),
    "instruction": query.strip(),
    "answer": answer.strip(),
    "short_answer": answer.strip(),
    "long_answer": "",
    "answer_start_token": None,  # MS MARCO n'a pas de positions
    "answer_end_token": None,
    "image": image,  # ← IMAGE PRÉ-GÉNÉRÉE
    "quality": quality,
    "estimated_height": estimated_height,
    "token_count": token_count,
    "extracted_token_count": token_count,  # Même que token_count (pas d'extraction)
    "metadata": {
        "source": "ms_marco",
        "config": config,
        "original_index": idx,
        "has_short": True,
        "has_long": False,
        "answer_type": "short",
        "extraction_method": "full_document",
        "generation_resolution": target_resolution,
        "quality_description": quality_desc,
    },
})
```

---

### 4. `src/deepsynth/data/instruction_dataset.py`

**AUCUN changement majeur nécessaire !**

Le `__getitem__()` actuel gère déjà bien:

```python
def __getitem__(self, idx):
    sample = self.dataset[idx]

    # Si image existe (notre cas), la charger
    if "image" in sample and sample["image"] is not None:
        image = sample["image"]

        # Appliquer transform (resize si besoin)
        if self.transform is not None:
            image = self.transform(image)
```

**Optionnel: Ajouter logging pour debug**

```python
# Dans __init__()
if len(self.dataset) > 0:
    first_sample = self.dataset[0]
    if "generation_resolution" in first_sample.get("metadata", {}):
        gen_res = first_sample["metadata"]["generation_resolution"]
        LOGGER.info(f"Dataset generated at resolution: {gen_res}")
```

---

### 5. `src/deepsynth/data/dataset_converters/__init__.py`

**Déjà à jour** (FiQA supprimé)

Pas de changements nécessaires.

---

## 📐 Algorithmes Détaillés

### Algorithme 1: Calcul Context Window Optimal

```
INPUT: target_resolution (string: "tiny"|"small"|"base"|"large"|"gundam")
OUTPUT: context_window (int: nombre de tokens)

1. Récupérer hauteur cible:
   resolutions = {"tiny": 512, "small": 640, "base": 1024, "large": 1280, "gundam": 1600}
   target_height = resolutions[target_resolution]

2. Calculer capacité en tokens:
   lines_per_image = target_height / 20  (20px par ligne)
   chars_per_image = lines_per_image * 100  (100 chars par ligne)
   tokens_per_image = chars_per_image / 5  (5 chars par token)

3. Context window = moitié de la capacité:
   context_window = tokens_per_image / 2

EXEMPLE:
  target_resolution = "gundam" (1600px)
  → lines = 1600/20 = 80 lignes
  → chars = 80*100 = 8000 chars
  → tokens = 8000/5 = 1600 tokens
  → context_window = 1600/2 = 800 tokens
```

### Algorithme 2: Décision Extraction Contextuelle

```
INPUT:
  - document_token_count (int)
  - target_resolution (string)
  - min_threshold (int, default=2000)

OUTPUT: (should_extract: bool, context_window: int)

1. Calculer context window optimal:
   context_window = calculate_optimal_context_window(target_resolution)

2. Calculer seuil d'extraction:
   extraction_threshold = max(min_threshold, context_window * 2)

   Raison: Si doc ≤ 2*window, l'extraction contextuelle ±window
           couvrirait presque tout le document de toute façon

3. Décision:
   if document_token_count <= extraction_threshold:
       return (False, 0)  // Garder document entier
   else:
       return (True, context_window)  // Extraction nécessaire

EXEMPLES:
  1. Doc 500 tokens, gundam (window=800)
     → threshold = max(2000, 800*2) = 2000
     → 500 ≤ 2000 → (False, 0) → Garder entier

  2. Doc 10000 tokens, gundam (window=800)
     → threshold = max(2000, 800*2) = 2000
     → 10000 > 2000 → (True, 800) → Extraction ±800

  3. Doc 1500 tokens, tiny (window=256)
     → threshold = max(2000, 256*2) = 2000
     → 1500 ≤ 2000 → (False, 0) → Garder entier
```

### Algorithme 3: Extraction Contextuelle avec Positions

```
INPUT:
  - tokens (list[str]): All document tokens
  - answer_start (int): Start position of answer
  - answer_end (int): End position of answer
  - context_window (int): Tokens to extract before/after

OUTPUT: extracted_text (str)

1. Calculer bornes:
   start_idx = max(0, answer_start - context_window)
   end_idx = min(len(tokens), answer_end + context_window)

2. Extraire:
   context_tokens = tokens[start_idx:end_idx]

3. Joindre:
   extracted_text = " ".join(context_tokens)

4. Return extracted_text

EXEMPLE:
  tokens = ["The", "Walking", "Dead", ..., "March", "18", "2018", ...]  (10000 tokens)
  answer_start = 3521
  answer_end = 3525
  context_window = 800

  → start_idx = max(0, 3521-800) = 2721
  → end_idx = min(10000, 3525+800) = 4325
  → context_tokens = tokens[2721:4325]  (1604 tokens)
  → extracted_text = "... March 18 2018 ..."
```

---

## 🧪 Tests de Validation

### Test 1: Vérifier Génération Image

```python
def test_image_generation_at_gundam():
    """Test que les images sont générées à résolution gundam."""
    from deepsynth.data.dataset_converters import convert_natural_questions

    dataset = convert_natural_questions(
        split="train",
        max_samples=10,
        streaming=True,
        target_resolution="gundam"
    )

    for sample in dataset:
        assert "image" in sample
        assert sample["image"] is not None

        # Vérifier résolution
        img = sample["image"]
        width, height = img.size
        assert width <= 1600, f"Width {width} > 1600"

        # Vérifier metadata
        assert sample["metadata"]["generation_resolution"] == "gundam"

    print("✅ Images générées à résolution gundam")
```

### Test 2: Vérifier Extraction Intelligente

```python
def test_intelligent_extraction():
    """Test que l'extraction se fait seulement si nécessaire."""
    from deepsynth.data.dataset_converters import convert_natural_questions

    dataset = convert_natural_questions(
        split="train",
        max_samples=100,
        streaming=True,
        target_resolution="gundam"
    )

    full_doc_count = 0
    contextual_count = 0

    for sample in dataset:
        method = sample["metadata"]["extraction_method"]
        token_count = sample["token_count"]
        extracted_count = sample["extracted_token_count"]

        if method == "full_document":
            full_doc_count += 1
            assert token_count == extracted_count, "Full doc should have same counts"
            assert token_count <= 2000, f"Full doc with {token_count} tokens should be extracted"

        elif method == "contextual":
            contextual_count += 1
            assert extracted_count < token_count, "Contextual should reduce token count"
            assert token_count > 2000, f"Doc with {token_count} tokens doesn't need extraction"

    print(f"✅ Full documents: {full_doc_count}")
    print(f"✅ Contextual extraction: {contextual_count}")
    assert full_doc_count > 0, "Should have some full documents"
    assert contextual_count > 0, "Should have some contextual extractions"
```

### Test 3: Vérifier Positions Stockées

```python
def test_answer_positions_stored():
    """Test que les positions des réponses sont stockées."""
    from deepsynth.data.dataset_converters import convert_natural_questions

    dataset = convert_natural_questions(
        split="train",
        max_samples=10,
        streaming=True,
        target_resolution="gundam"
    )

    for sample in dataset:
        if sample["metadata"]["has_long"] or sample["metadata"]["has_short"]:
            assert "answer_start_token" in sample
            assert "answer_end_token" in sample
            assert sample["answer_start_token"] is not None
            assert sample["answer_end_token"] is not None

    print("✅ Answer positions stored correctly")
```

### Test 4: Vérifier Document Complet Stocké

```python
def test_full_document_stored():
    """Test que le document COMPLET est stocké dans 'text'."""
    from deepsynth.data.dataset_converters import convert_natural_questions

    dataset = convert_natural_questions(
        split="train",
        max_samples=10,
        streaming=True,
        target_resolution="gundam"
    )

    for sample in dataset:
        text_tokens = len(sample["text"].split())
        stored_count = sample["token_count"]

        # Le nombre de tokens dans 'text' devrait matcher token_count
        # (avec petite marge pour variations de tokenization)
        assert abs(text_tokens - stored_count) < 100, \
            f"Text has {text_tokens} tokens but token_count is {stored_count}"

        # Si extraction contextuelle, extracted_token_count devrait être < token_count
        if sample["metadata"]["extraction_method"] == "contextual":
            assert sample["extracted_token_count"] < sample["token_count"]

    print("✅ Full document stored in 'text' field")
```

### Test 5: Benchmark Temps de Loading

```python
def test_loading_speed():
    """Test que le chargement est rapide (images pré-générées)."""
    import time
    from deepsynth.data.dataset_converters import convert_natural_questions
    from deepsynth.data.instruction_dataset import InstructionDataset

    # Convertir dataset
    raw_dataset = convert_natural_questions(
        split="train",
        max_samples=100,
        streaming=True,
        target_resolution="gundam"
    )

    # Créer InstructionDataset
    dataset = InstructionDataset(raw_dataset)

    # Mesurer temps de chargement
    start = time.time()
    for i in range(10):
        sample = dataset[i]
        assert sample["image"] is not None
    end = time.time()

    avg_time = (end - start) / 10 * 1000  # ms

    print(f"✅ Average loading time: {avg_time:.1f}ms per sample")
    assert avg_time < 50, f"Loading too slow: {avg_time}ms (should be <50ms)"
```

---

## ✅ Critères d'Acceptation

### Fonctionnel

- [ ] Images pré-générées à résolution gundam (1600px)
- [ ] Extraction contextuelle intelligente (seulement si doc > 2000 tokens)
- [ ] Document COMPLET stocké dans field "text"
- [ ] Positions answer stockées (answer_start_token, answer_end_token)
- [ ] Long answer en priorité
- [ ] Metadata complet avec generation_resolution
- [ ] MS MARCO également avec images pré-générées
- [ ] FiQA supprimé

### Performance

- [ ] Chargement image <50ms par sample
- [ ] Training overhead 0% (pas de génération à la volée)
- [ ] Dataset size acceptable sur HuggingFace

### Qualité

- [ ] Natural Questions: 40% skip rate (normal - pas de réponse)
- [ ] Quality distribution: 80%+ excellent/good
- [ ] Images lisibles à résolution gundam
- [ ] Downscale vers tiny/base fonctionne

### Tests

- [ ] Tous les tests unitaires passent
- [ ] Test intégration génération dataset
- [ ] Test training avec dataset généré
- [ ] Validation qualité visuelle des images

---

## 📊 Estimation Taille Dataset

### Natural Questions (300k samples)

**Avec extraction contextuelle** (docs réduits à ~1500 tokens avg):
- Image size: 1600px × 1500px × 3 (RGB) ≈ 7.2 MB par sample
- Total: 300,000 × 7.2 MB ≈ **2.16 TB**

**Compression HuggingFace** (PNG avec compression):
- Facteur compression: ~5-10x
- Total compressé: **216-432 GB**

### MS MARCO (1M samples)

**Documents courts** (~200 tokens avg):
- Image size: 1600px × 200px × 3 ≈ 960 KB par sample
- Total: 1,000,000 × 960 KB ≈ **960 GB**
- Compressé: **96-192 GB**

### Total Estimé

- Natural Questions: **216-432 GB**
- MS MARCO: **96-192 GB**
- **TOTAL: 312-624 GB** sur HuggingFace

C'est gérable pour HuggingFace (ils supportent des datasets bien plus larges).

---

## 🚀 Ordre d'Implémentation Recommandé

### Phase 1: Fonctions Utilitaires (1-2h)
1. `quality_calculator.py`:
   - `calculate_optimal_context_window()`
   - `should_extract_context()`

### Phase 2: Natural Questions (2-3h)
1. Modifier `convert_natural_questions()`:
   - Ajouter paramètre `target_resolution`
   - Calcul context window
   - Décision extraction intelligente
   - Générer image
   - Stocker positions + document complet

### Phase 3: MS MARCO (1h)
1. Modifier `convert_ms_marco()`:
   - Ajouter paramètre `target_resolution`
   - Générer image
   - Metadata

### Phase 4: Tests (1-2h)
1. Tests unitaires
2. Tests intégration
3. Validation visuelle

### Phase 5: Documentation (30min)
1. Update README
2. Update tests docs

**TOTAL ESTIMÉ: 6-9 heures de développement**

---

## 💡 Notes pour le Développeur

### Pièges à Éviter

1. **Ne PAS utiliser `[:1000]` pour tronquer le texte**
   - Utiliser `should_extract_context()` pour décider
   - Utiliser `extract_contextual_window()` pour extraire proprement

2. **Stocker le document COMPLET dans "text"**
   - Pas le document extrait
   - L'extraction est seulement pour l'image

3. **Positions en tokens, pas en caractères**
   - `answer_start_token` et `answer_end_token`
   - Pas `answer_start_char`

4. **Image resolution = gundam (1600px width)**
   - Pas 1024 (base) ou 512 (tiny)
   - gundam = meilleure qualité

### Optimisations Futures

- Caching des images générées en local pour dev
- Génération parallèle (multiprocessing)
- Validation qualité automatique des images
- Support multi-résolution (stocker tiny + gundam)

---

## 📞 Questions / Clarifications

Si des questions pendant l'implémentation, vérifier:
1. Ce document en premier
2. Code existant dans `natural_questions.py` (pour patterns)
3. Tests existants (pour exemples)

**Contact**: [Ton contact ici]

---

**Document Version**: 1.0
**Date**: 2025-10-28
**Approuvé par**: Loic
**Résolution cible**: gundam (1600px)
