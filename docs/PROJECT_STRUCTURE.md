# Structure du Projet DeepSynth

Ce document décrit l'organisation complète du projet après restructuration.

## 📁 Vue d'ensemble

```
DeepSynth/
├── 📄 Fichiers de configuration (racine)
│   ├── README.md              # Documentation principale
│   ├── pyproject.toml         # Configuration packaging Python
│   ├── requirements.txt       # Dépendances Python
│   ├── setup.sh              # Script d'installation
│   ├── sitecustomize.py      # Configuration Python path
│   ├── Makefile              # Commandes rapides
│   ├── .env.example          # Template configuration
│   └── .gitignore            # Fichiers ignorés par git
│
├── 📦 src/                   # Code source principal
│   ├── deepsynth/            # Package Python principal
│   │   ├── config/           # Gestion configuration
│   │   ├── data/             # Traitement des données
│   │   ├── evaluation/       # Métriques et benchmarks
│   │   ├── inference/        # Inférence et API
│   │   ├── pipelines/        # Pipelines de traitement
│   │   └── training/         # Modules d'entraînement
│   │
│   └── apps/                 # Applications
│       └── web/              # Interface web
│           ├── config.py
│           └── ui/           # Templates et assets
│
├── 🧪 tests/                 # Tests organisés
│   ├── data/                 # Tests transformations
│   ├── pipelines/            # Tests pipelines
│   ├── system/               # Tests système
│   └── training/             # Tests entraînement
│
├── 🔧 scripts/               # Scripts et outils
│   ├── cli/                  # Scripts CLI principaux
│   ├── maintenance/          # Outils de maintenance
│   └── shell/                # Scripts shell
│
├── 📚 docs/                  # Documentation
│   ├── reports/              # Rapports de tests
│   ├── architecture/         # Documentation architecture
│   └── *.md                  # Guides divers
│
└── 🚀 deploy/                # Déploiement
    ├── Dockerfile
    ├── docker-compose.yml
    └── scripts/              # Scripts de déploiement
```

## 🎯 Règles d'organisation

### Racine du projet
**Seulement les fichiers essentiels :**
- Configuration (pyproject.toml, requirements.txt, .env.example)
- Documentation (README.md)
- Scripts d'installation (setup.sh)
- Configuration Python (sitecustomize.py)
- Outils de développement (Makefile)

### Code source (src/)
Tout le code Python réutilisable doit être dans `src/deepsynth/` ou `src/apps/`

### Scripts (scripts/)
- `scripts/cli/` : Scripts Python CLI principaux
- `scripts/shell/` : Scripts shell pour automatisation
- `scripts/maintenance/` : Outils de maintenance

### Tests (tests/)
Structure miroir de `src/deepsynth/` :
- `tests/data/` → tests pour `src/deepsynth/data/`
- `tests/pipelines/` → tests pour `src/deepsynth/pipelines/`
- etc.

### Documentation (docs/)
- Guides utilisateur
- Documentation technique
- Rapports dans `docs/reports/`

## 🚀 Commandes rapides

```bash
# Via Makefile
make setup          # Installation
make test           # Tests
make pipeline       # Pipeline complet
make web            # Interface web

# Directement
python scripts/run_benchmark.py
python scripts/run_complete_multilingual_pipeline.py
python -m src.apps.web
```

## 📦 Import des modules

```python
# Import depuis src/deepsynth/
from deepsynth.config import Config
from deepsynth.data import TextToImageConverter
from deepsynth.pipelines import run_incremental_pipeline
from deepsynth.evaluation.benchmarks import list_benchmark_datasets

# Import depuis src/apps/
from apps.web.ui.state_manager import StateManager
```

## 🔄 Migration depuis l'ancienne structure

### Anciens scripts à la racine
Les anciens scripts ont été déplacés :
- ✅ `run_*.py` → `scripts/run_*.py`
- ✅ `test_*.py` → `tests/*/test_*.py`
- ✅ `evaluation/` → `src/deepsynth/evaluation/`
- ✅ `web_ui/` → `src/apps/web/`

### Compatibilité
Utilisez le Makefile pour une compatibilité maximale :
```bash
# Ancien: python run_benchmark.py
# Nouveau: make benchmark  OU  python scripts/run_benchmark.py
```

## 📝 Conventions de nommage

- **Packages** : `snake_case` (ex: `deepsynth.data`)
- **Modules** : `snake_case.py` (ex: `text_to_image.py`)
- **Classes** : `PascalCase` (ex: `TextToImageConverter`)
- **Fonctions** : `snake_case` (ex: `run_pipeline`)
- **Constantes** : `UPPER_CASE` (ex: `MAX_SAMPLES`)

## 🏗️ Principes d'architecture

1. **Séparation des responsabilités** : Code source ≠ Tests ≠ Scripts ≠ Documentation
2. **Réutilisabilité** : Tout le code réutilisable dans `src/`
3. **Testabilité** : Structure de tests miroir du code source
4. **Clarté** : Racine minimaliste, organisation logique
5. **Standards Python** : Respect de PEP 8 et structure src layout
