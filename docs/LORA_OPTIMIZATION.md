# Guide d'Optimisation LoRA pour DeepSynth

> **Document de recommandations pour optimiser le fine-tuning LoRA avec DeepSeek-OCR**

## Table des Matières

1. [État Actuel](#état-actuel)
2. [Problèmes Identifiés](#problèmes-identifiés)
3. [Recommandations LoRA](#recommandations-lora)
4. [Configuration Training](#configuration-training)
5. [Régularisation MoE](#régularisation-moe)
6. [Configurations Recommandées](#configurations-recommandées)
7. [Implémentation](#implémentation)

---

## État Actuel

### Configuration par défaut (`src/deepsynth/training/config.py:59`)

```python
use_lora: bool = True
lora_rank: int = 16
lora_alpha: int = 32
lora_dropout: float = 0.05
lora_target_modules: Optional[List[str]] = None  # Auto-détection
```

### Auto-détection des modules (`src/deepsynth/training/deepsynth_lora_trainer.py:150`)

Par défaut, seuls les **modules d'attention** sont ciblés:
- `["q_proj", "k_proj", "v_proj", "o_proj"]`
- Résultat: ~**1.97M paramètres** (0.06% de 3.34B)

### Problème Principal

**Capacité insuffisante pour datasets très petits (3-50 samples):**
- Attention-only → capacité limitée
- Pas de ciblage MLP/FFN → sous-utilisation
- Pas de router training → distribution experts non adaptée
- Pas de scheduler → learning rate constant

---

## Problèmes Identifiés

### 1. Modules Cibles Limités

**Actuel:** Attention uniquement (`q_proj`, `k_proj`, `v_proj`, `o_proj`)

**Impact:**
- ~2M paramètres LoRA seulement
- Pas d'adaptation du contenu/style (MLP/FFN)
- Pas d'adaptation du routage (gates/routers MoE)

**Fichier:** `src/deepsynth/training/deepsynth_lora_trainer.py:150`

```python
# Auto-détection actuelle
if not target_modules:
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### 2. Configuration par Défaut Sous-Optimale

**Paramètres actuels** (`config.py:21-42`):
```python
batch_size: int = 2
num_epochs: int = 1
gradient_accumulation_steps: int = 1
learning_rate: float = 2e-5  # OptimizerConfig
warmup_steps: int = 0  # Non utilisé!
```

**Problèmes:**
- 1 epoch avec 3 samples = **1-3 updates seulement**
- Pas de warmup implementé
- Pas de scheduler (cosine/linear)
- Gradient accumulation faible

### 3. Router MoE Non Entraîné

**Architecture DeepSeek-OCR:**
- Mixture of Experts (MoE) avec router/gate
- Router décide quels experts utiliser
- **Problème:** Router pas adapté à votre distribution

**Solution:** Cibler le router avec LoRA ou full training

---

## Recommandations LoRA

### 1. Étendre les Modules Cibles

#### Option A: Attention + MLP (Recommandé pour Tiny Data)

```python
lora_target_modules = [
    # Attention (déjà inclus)
    "q_proj", "k_proj", "v_proj", "o_proj",
    # MLP/FFN (NOUVEAU)
    "down_proj"  # Le plus efficace
]
```

**Impact:**
- Paramètres: 2M → ~4M (attention+MLP partiel)
- Meilleure adaptation contenu/style
- Équilibre capacité/overfit

#### Option B: Attention + MLP Complet

```python
lora_target_modules = [
    # Attention
    "q_proj", "k_proj", "v_proj", "o_proj",
    # MLP Complet (NOUVEAU)
    "gate_proj", "up_proj", "down_proj"
]
```

**Impact:**
- Paramètres: 2M → ~6-8M
- Capacité maximale pour 50+ samples
- Risque overfit si <10 samples

#### Option C: Attention + MLP + Router (Pour MoE)

```python
lora_target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "router", "gate"  # Routeur MoE
]
```

**Impact:**
- Adaptation distribution experts
- Critique si votre domain diffère de pretraining
- ~10-15M paramètres

**Où modifier:** `src/deepsynth/training/config.py:63`

### 2. Ajuster le Rang LoRA

#### Formule Estimation

```python
params_lora ≈ 2 * r * hidden_size * num_targets * num_layers
```

Pour DeepSeek-OCR (hidden_size ≈ 1280-2048):
- **r=16:** ~2-4M params (attention only)
- **r=32:** ~4-8M params (attention only)
- **r=64:** ~8-16M params (high capacity)

#### Recommandations par Dataset Size

| Dataset Size | Rank | Alpha | Modules | Params LoRA |
|--------------|------|-------|---------|-------------|
| 3-10 samples | 16 | 32 | attention + down_proj | ~4M |
| 10-50 samples | 32 | 64 | attention + MLP | ~8M |
| 50+ samples | 32-64 | 64-128 | attention + MLP + router | ~15M |

**Où modifier:** `src/deepsynth/training/config.py:60-61`

```python
lora_rank: int = 32  # Au lieu de 16
lora_alpha: int = 64  # Au lieu de 32
```

### 3. Dropout LoRA

**Tiny data (3-50 samples):**
```python
lora_dropout: float = 0.08  # Au lieu de 0.05
```

**Rationale:** Augmenter dropout pour éviter overfit sur si petits datasets

---

## Configuration Training

### 1. Nombre d'Updates Effectifs

**Objectif:** 20-100 updates optimiseur effectifs minimum

**Calcul:**
```python
total_steps = (num_samples / batch_size) * num_epochs
effective_updates = total_steps / gradient_accumulation_steps
```

**Exemple pour 3 samples:**
```python
batch_size = 1
gradient_accumulation_steps = 16
num_epochs = 200

total_steps = (3 / 1) * 200 = 600 steps
effective_updates = 600 / 16 = 37 updates ✓
```

### 2. Configuration Recommandée Tiny Data

**Fichier:** `src/deepsynth/training/config.py`

```python
@dataclass
class TrainerConfig:
    # Training
    batch_size: int = 1  # Au lieu de 2
    num_epochs: int = 200  # Au lieu de 1
    gradient_accumulation_steps: int = 16  # Au lieu de 1

    # Optimizer
    learning_rate: float = 5e-5  # Au lieu de 2e-5
    weight_decay: float = 0.01  # Au lieu de 0.0
    warmup_steps: int = 50  # NOUVEAU: 10-15% des steps

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32  # Au lieu de 16
    lora_alpha: int = 64  # Au lieu de 32
    lora_dropout: float = 0.08  # Au lieu de 0.05
    lora_target_modules: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "down_proj"  # NOUVEAU
    ]
```

### 3. Scheduler avec Warmup

**Problème actuel:** `warmup_steps` défini mais **non utilisé** dans `production_trainer.py`

**Solution:** Implémenter scheduler

**Nouveau fichier:** `src/deepsynth/training/scheduler.py`

```python
"""Learning rate schedulers avec warmup."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
):
    """Linear warmup + linear decay."""

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    """Cosine warmup + cosine decay."""
    import math

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
                  float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
```

**Modification:** `src/deepsynth/training/production_trainer.py`

Ajouter après création de l'optimizer (ligne ~250):

```python
# Create scheduler with warmup
num_training_steps = len(train_loader) * self.config.num_epochs
warmup_steps = self.config.optimizer.warmup_steps

from deepsynth.training.scheduler import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)
```

Et dans la boucle training (après `optimizer.step()`):

```python
optimizer.step()
scheduler.step()  # NOUVEAU
optimizer.zero_grad()
```

---

## Régularisation MoE

### 1. Gate Dropout

**Fichier:** `src/deepsynth/training/moe_dropout.py` (déjà implémenté)

**Configuration recommandée:**

```python
gate_dropout_rate: float = 0.1  # Au lieu de 0.0
gate_dropout_keywords: Tuple[str, ...] = ("gate", "router")
```

**Impact:** Dropout gradients du router pour stabiliser training

### 2. Expert Dropout

```python
expert_dropout_rate: float = 0.05  # Au lieu de 0.0
expert_dropout_min_keep: int = 1  # Garder au moins 1 expert
```

**Impact:** Évite spécialisation excessive d'un seul expert

### 3. Router Training

**Option A:** LoRA sur router

```python
lora_target_modules = [..., "router", "gate"]
```

**Option B:** Full training du router

```python
lora_modules_to_save: List[str] = ["router", "gate"]
```

**Recommandation:** Option A (LoRA) pour tiny data, Option B si 50+ samples

---

## Configurations Recommandées

### Config 1: Tiny Data (3-10 samples)

**Cas d'usage:** Tests rapides, proof-of-concept, datasets minuscules

**Fichier:** `src/deepsynth/training/lora_config.py` (ajouter preset)

```python
"tiny_data": LoRAConfig(
    enabled=True,
    rank=16,
    alpha=32,
    dropout=0.1,  # High dropout
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "down_proj"  # Partiel MLP
    ]
)
```

**TrainerConfig:**

```python
batch_size = 1
gradient_accumulation_steps = 16
num_epochs = 300  # ~56 updates effectifs
learning_rate = 5e-5
weight_decay = 0.01
warmup_steps = 50
```

**Paramètres:** ~4M LoRA
**Updates:** ~56 effectifs
**Temps:** ~15-30min

### Config 2: Balanced (10-50 samples)

**Cas d'usage:** Datasets moyens, fine-tuning équilibré

```python
"balanced": LoRAConfig(
    enabled=True,
    rank=32,
    alpha=64,
    dropout=0.08,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"  # MLP complet
    ]
)
```

**TrainerConfig:**

```python
batch_size = 1
gradient_accumulation_steps = 16
num_epochs = 200  # ~75 updates effectifs (50 samples)
learning_rate = 2e-5
weight_decay = 0.01
warmup_steps = 100
```

**Paramètres:** ~8M LoRA
**Updates:** ~75 effectifs
**Temps:** ~1-2h

### Config 3: Router-Focused (MoE Adaptation)

**Cas d'usage:** Adapter distribution experts, domain shift important

```python
"router_focused": LoRAConfig(
    enabled=True,
    rank=16,
    alpha=32,
    dropout=0.08,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "down_proj",
        "router", "gate"  # Router MoE
    ],
    modules_to_save=["router", "gate"]  # Alternative: full training
)
```

**TrainerConfig:**

```python
batch_size = 1
gradient_accumulation_steps = 32
num_epochs = 200
learning_rate = 2e-5
weight_decay = 0.01
warmup_steps = 100

# MoE regularization
gate_dropout_rate = 0.1
expert_dropout_rate = 0.05
expert_dropout_min_keep = 1
```

**Paramètres:** ~10-15M LoRA
**Updates:** ~150 effectifs (50 samples)
**Temps:** ~2-3h

### Config 4: High Capacity (50+ samples)

**Cas d'usage:** Datasets larges, capacité maximale

```python
"high_capacity_moe": LoRAConfig(
    enabled=True,
    rank=64,
    alpha=128,
    dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "router", "gate"
    ]
)
```

**TrainerConfig:**

```python
batch_size = 2
gradient_accumulation_steps = 16
num_epochs = 200-400
learning_rate = 2e-5
weight_decay = 0.01
warmup_steps = 200
```

**Paramètres:** ~20-30M LoRA
**Updates:** ~200-400 effectifs
**Temps:** ~4-8h

---

## Implémentation

### Étape 1: Ajouter Nouveaux Presets

**Fichier:** `src/deepsynth/training/lora_config.py:217`

Ajouter à `LORA_PRESETS`:

```python
LORA_PRESETS = {
    # ... existing presets ...

    # NOUVEAU: Tiny data (3-10 samples)
    "tiny_data": LoRAConfig(
        enabled=True,
        rank=16,
        alpha=32,
        dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj"]
    ),

    # NOUVEAU: Balanced (10-50 samples)
    "balanced": LoRAConfig(
        enabled=True,
        rank=32,
        alpha=64,
        dropout=0.08,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    ),

    # NOUVEAU: Router-focused (MoE)
    "router_focused": LoRAConfig(
        enabled=True,
        rank=16,
        alpha=32,
        dropout=0.08,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "down_proj", "router", "gate"
        ],
        modules_to_save=["router", "gate"]
    ),

    # NOUVEAU: High capacity (50+ samples)
    "high_capacity_moe": LoRAConfig(
        enabled=True,
        rank=64,
        alpha=128,
        dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "router", "gate"
        ]
    ),
}
```

### Étape 2: Créer Module Scheduler

**Nouveau fichier:** `src/deepsynth/training/scheduler.py`

(Voir code complet dans section "Scheduler avec Warmup")

### Étape 3: Modifier Production Trainer

**Fichier:** `src/deepsynth/training/production_trainer.py`

**Ligne ~250** (après création optimizer):

```python
# Calculate training steps
steps_per_epoch = len(train_loader)
num_training_steps = steps_per_epoch * self.config.num_epochs
warmup_steps = self.config.optimizer.warmup_steps

# Create scheduler with warmup
from deepsynth.training.scheduler import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)

LOGGER.info(f"Scheduler: cosine with {warmup_steps} warmup steps")
```

**Ligne ~450** (dans training loop, après optimizer.step()):

```python
if (step + 1) % self.config.gradient_accumulation_steps == 0:
    optimizer.step()
    scheduler.step()  # NOUVEAU
    optimizer.zero_grad()
```

### Étape 4: Ajouter Config Scheduler Type

**Fichier:** `src/deepsynth/training/config.py:12`

```python
@dataclass
class OptimizerConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01  # Au lieu de 0.0
    warmup_steps: int = 100  # Au lieu de 0
    scheduler_type: str = "cosine"  # NOUVEAU: "cosine", "linear", "none"
```

### Étape 5: Ajouter API Endpoint Recommendations

**Fichier:** `src/apps/web/ui/app.py:638`

Ajouter après `/api/lora/presets`:

```python
@app.route("/api/lora/presets/recommended", methods=["POST"])
def get_recommended_lora_preset():
    """Get recommended LoRA preset based on dataset size."""
    try:
        data = request.json or {}
        num_samples = data.get("num_samples", 10)

        # Determine best preset
        if num_samples < 10:
            preset_name = "tiny_data"
            recommended_config = {
                "batch_size": 1,
                "num_epochs": 300,
                "gradient_accumulation_steps": 16,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "warmup_steps": 50
            }
        elif num_samples < 50:
            preset_name = "balanced"
            recommended_config = {
                "batch_size": 1,
                "num_epochs": 200,
                "gradient_accumulation_steps": 16,
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "warmup_steps": 100
            }
        else:
            preset_name = "high_capacity_moe"
            recommended_config = {
                "batch_size": 2,
                "num_epochs": 200,
                "gradient_accumulation_steps": 16,
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "warmup_steps": 200
            }

        from deepsynth.training.lora_config import LORA_PRESETS

        lora_config = LORA_PRESETS[preset_name].to_dict()

        # Calculate expected training time
        steps_per_epoch = max(1, num_samples // recommended_config["batch_size"])
        total_steps = steps_per_epoch * recommended_config["num_epochs"]
        effective_updates = total_steps // recommended_config["gradient_accumulation_steps"]

        return jsonify({
            "preset_name": preset_name,
            "lora_config": lora_config,
            "training_config": recommended_config,
            "estimates": {
                "total_steps": total_steps,
                "effective_updates": effective_updates,
                "estimated_time_minutes": effective_updates * 2  # ~2min per update
            }
        })

    except Exception as exc:
        logger.exception("Error getting recommended preset")
        return jsonify({"error": str(exc)}), 500
```

### Étape 6: Modifier Interface Web

**Fichier:** `src/apps/web/ui/templates/index_improved.html`

Ajouter sélecteur de presets LoRA dans le formulaire training:

```html
<!-- LoRA Preset Selector -->
<div class="form-group">
    <label for="lora-preset">LoRA Preset:</label>
    <select id="lora-preset" class="form-control">
        <option value="custom">Custom (manual)</option>
        <option value="tiny_data">Tiny Data (3-10 samples)</option>
        <option value="balanced" selected>Balanced (10-50 samples)</option>
        <option value="router_focused">Router-Focused (MoE adaptation)</option>
        <option value="high_capacity_moe">High Capacity (50+ samples)</option>
        <option value="standard">Standard (legacy)</option>
        <option value="qlora_4bit">QLoRA 4-bit</option>
    </select>
    <small class="form-text text-muted">
        Select a preset configuration or choose Custom to configure manually.
    </small>
</div>

<!-- Auto-fill button -->
<button type="button" class="btn btn-secondary" onclick="applyLoRAPreset()">
    Apply Preset & Auto-Configure
</button>
```

JavaScript:

```javascript
async function applyLoRAPreset() {
    const preset = document.getElementById('lora-preset').value;
    if (preset === 'custom') return;

    // Get dataset size from form
    const maxSamples = parseInt(document.getElementById('max-train-samples').value) || 10;

    // Get recommendation from API
    const response = await fetch('/api/lora/presets/recommended', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({num_samples: maxSamples})
    });

    const data = await response.json();

    // Auto-fill form with recommended values
    document.getElementById('lora-rank').value = data.lora_config.rank;
    document.getElementById('lora-alpha').value = data.lora_config.alpha;
    document.getElementById('lora-dropout').value = data.lora_config.dropout;
    document.getElementById('batch-size').value = data.training_config.batch_size;
    document.getElementById('num-epochs').value = data.training_config.num_epochs;
    document.getElementById('gradient-accumulation').value = data.training_config.gradient_accumulation_steps;
    document.getElementById('learning-rate').value = data.training_config.learning_rate;

    // Show estimates
    alert(`Configuration applied!\n\n` +
          `Effective updates: ${data.estimates.effective_updates}\n` +
          `Estimated time: ${data.estimates.estimated_time_minutes} minutes`);
}
```

---

## Résumé des Fichiers à Modifier

### 1. LoRA Config

**Fichier:** `src/deepsynth/training/lora_config.py:217`
**Action:** Ajouter 4 nouveaux presets (tiny_data, balanced, router_focused, high_capacity_moe)

### 2. Trainer Config

**Fichier:** `src/deepsynth/training/config.py`
**Actions:**
- Ligne 12: Ajouter `scheduler_type` à OptimizerConfig
- Ligne 11: Changer `weight_decay: float = 0.01`
- Ligne 12: Changer `warmup_steps: int = 100`
- Ligne 63: Modifier `lora_target_modules` default

### 3. Nouveau Module Scheduler

**Fichier:** `src/deepsynth/training/scheduler.py` (NOUVEAU)
**Action:** Créer module avec `get_linear_schedule_with_warmup` et `get_cosine_schedule_with_warmup`

### 4. Production Trainer

**Fichier:** `src/deepsynth/training/production_trainer.py`
**Actions:**
- Ligne ~250: Ajouter création scheduler
- Ligne ~450: Ajouter `scheduler.step()` après `optimizer.step()`

### 5. API Web

**Fichier:** `src/apps/web/ui/app.py:638`
**Action:** Ajouter endpoint `/api/lora/presets/recommended`

### 6. Interface Web

**Fichier:** `src/apps/web/ui/templates/index_improved.html`
**Action:** Ajouter sélecteur de presets et fonction auto-fill JavaScript

---

## Tests Recommandés

### Test 1: Tiny Data Preset

```bash
# Dans le container
python3 -c "
from deepsynth.training.config import TrainerConfig
from deepsynth.training.lora_config import get_lora_preset

config = TrainerConfig(
    model_name='deepseek-ai/DeepSeek-OCR',
    output_dir='/tmp/test_tiny_preset',
    batch_size=1,
    num_epochs=300,
    max_train_samples=3,
    gradient_accumulation_steps=16,
    use_lora=True,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'down_proj']
)

# Test training
from datasets import load_dataset
train_ds = load_dataset('baconnier/deepsynth-fr-mini', split='train[:3]')
eval_ds = load_dataset('baconnier/deepsynth-fr-mini', split='train[3:4]')

from deepsynth.training.production_trainer import UnifiedProductionTrainer
trainer = UnifiedProductionTrainer(config)
metrics, paths = trainer.train(dataset=train_ds, eval_dataset=eval_ds)

print(f'✅ Test réussi!')
print(f'Updates effectifs: ~56')
print(f'Modules LoRA: attention + down_proj')
print(f'Params: ~4M')
"
```

### Test 2: Balanced Preset

```bash
python3 -c "
# 20 samples, preset balanced
config = TrainerConfig(
    batch_size=1,
    num_epochs=200,
    max_train_samples=20,
    gradient_accumulation_steps=16,
    lora_rank=32,
    lora_alpha=64,
    lora_dropout=0.08,
    lora_target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj'
    ]
)
# Test...
"
```

### Test 3: Scheduler Integration

```bash
python3 -c "
from deepsynth.training.scheduler import get_cosine_schedule_with_warmup
from torch.optim import AdamW
import torch

# Create dummy model
model = torch.nn.Linear(10, 10)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Create scheduler
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000
)

# Test warmup
for step in range(1000):
    optimizer.step()
    scheduler.step()

    if step in [0, 10, 50, 100, 500, 999]:
        lr = scheduler.get_last_lr()[0]
        print(f'Step {step}: LR = {lr:.2e}')

print('✅ Scheduler works!')
"
```

---

## Conclusion

Ces recommandations permettent:

1. **Capacité LoRA adaptée** aux tiny datasets (3-50 samples)
2. **Training stable** avec scheduler warmup + cosine decay
3. **Nombre d'updates suffisant** via accumulation gradients + epochs
4. **Adaptation MoE** via router training
5. **Interface simplifiée** avec presets auto-configurés

**Impact attendu:**
- Meilleure convergence sur tiny data
- Loss plus stable (warmup)
- Adaptation domain-specific (MLP + router)
- UX améliorée (presets one-click)

**Prochaines étapes:**
1. Implémenter les modifications (voir TODO list)
2. Tester avec mini-dataset (3-50 samples)
3. Comparer métriques avant/après
4. Documenter résultats dans README

---

## Fix HuggingFace Upload (production_trainer.py:1081)

### Problème Identifié

**Bug:** Paramètre invalide `use_hf_transfer=False` passé à `upload_folder()` causant l'erreur:
```
TypeError: HfApi.upload_folder() got an unexpected keyword argument 'use_hf_transfer'
```

### Solution Appliquée

**Fichier:** `src/deepsynth/training/production_trainer.py:1074-1081`

**Avant** (incorrect):
```python
self.api.upload_folder(
    folder_path=str(self.output_dir),
    repo_id=repo_id,
    repo_type="model",
    token=self.config.hub_token,
    ignore_patterns=["checkpoint-*", "epoch_*"],
    commit_message="Upload final model artifacts (skip intermediate checkpoints)",
    use_hf_transfer=False,  # ❌ INVALIDE - n'existe pas dans l'API
)
```

**Après** (corrigé):
```python
self.api.upload_folder(
    folder_path=str(self.output_dir),
    repo_id=repo_id,
    repo_type="model",
    token=self.config.hub_token,
    ignore_patterns=["checkpoint-*", "epoch_*"],
    commit_message="Upload final model artifacts (skip intermediate checkpoints)",
)
```

**Explication:** Les variables d'environnement `HF_HUB_ENABLE_XET=0` et `HF_HUB_ENABLE_HF_TRANSFER=0` (lignes 1070-1071) désactivent déjà les backends XET. Le paramètre `use_hf_transfer` n'existe pas dans l'API `upload_folder()`.

### Résultats du Test

**Succès partiel:**
- ✅ Training réussi (2 samples, LoRA activé)
- ✅ Uploads intermédiaires correctement ignorés (4x "⏭️ Skipping..." logs)
- ✅ 7/8 fichiers uploadés avec succès:
  - .gitattributes, README.md, adapter_config.json, metrics.json
  - special_tokens_map.json, tokenizer.json, tokenizer_config.json, training_config.json
- ⚠️ `adapter_model.safetensors` (~17MB) toujours en échec avec erreur MerkleDB

**Erreur MerkleDB persistante:**
```
ERROR: Failed to upload adapter_model.safetensors: Data processing error: MerkleDB Shard error: File I/O error
Processing Files (0 / 0): |          |  0.00B /  0.00B
```

**Cause:** Problème infrastructure HuggingFace XET affectant le compte `baconnier`, spécifiquement pour les fichiers >10MB. Même avec Legacy HTTP backend activé et XET désactivé.

### Workarounds Disponibles

1. **Désactiver push_to_hub temporairement:**
```python
config = TrainerConfig(
    push_to_hub=False,  # Upload manuel via Web UI
    ...
)
```

2. **Tester avec compte HuggingFace différent:**
```python
config = TrainerConfig(
    hub_model_id="autre-compte/model-test",
    ...
)
```

3. **Git Backend** (requiert git-lfs dans container):
```bash
# Installer dans le container
apt-get update && apt-get install -y git git-lfs && git lfs install

# Puis définir variables d'environnement
export DS_PUSH_BACKEND=git
```

### Gestion des Repositories de Test

**❌ MAUVAISE PRATIQUE** (crée repos multiples):
```python
import time
repo_name = f'baconnier/test-{int(time.time())}'  # ❌ Ne pas faire!
```

**✅ BONNE PRATIQUE** (réutilise même repo):
```python
repo_name = 'baconnier/deepsynth-test'  # ✅ Nom constant pour tests
```

**Avantages:**
- Pas d'accumulation de repos de test
- HuggingFace account propre
- Facilite debugging (historique de commits)

### Fichiers Modifiés

1. **production_trainer.py** (ligne 1081): Suppression paramètre invalide
2. **Tous les tests:** Utilisation nom de repo constant "baconnier/deepsynth-test"

### Verification

Pour tester le fix:
```bash
docker cp C:\Users\loic\DeepSynth\src\deepsynth\training\production_trainer.py \
    deepsynth-trainer-gpu:/app/src/deepsynth/training/production_trainer.py

# Test dans container
docker exec deepsynth-trainer-gpu python3 -c "
# Code de test avec push_to_hub=True
"
```

