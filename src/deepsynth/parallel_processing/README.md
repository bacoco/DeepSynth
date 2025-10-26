# Parallel Processing Module

Ce module permet le traitement parallèle des datasets DeepSynth pour accélérer considérablement la génération des datasets.

## Structure

```
src/deepsynth/parallel_processing/
├── __init__.py                    # Module Python
├── parallel_datasets_builder.py  # Logique principale de parallélisation
├── run_parallel_datasets.py      # Interface utilisateur
└── README.md                     # Cette documentation
```

## Utilisation

### Depuis le répertoire racine
```bash
python run_parallel_processing.py
```

### Depuis ce répertoire
```bash
PYTHONPATH=src python -m deepsynth.parallel_processing.run_parallel_datasets
```

### En tant que module Python
```python
from deepsynth.parallel_processing import ParallelDatasetsBuilder

builder = ParallelDatasetsBuilder(max_workers=3)
results = builder.run_parallel_processing()
```

## Fonctionnalités

- ✅ **Traitement parallèle** : Plusieurs datasets traités simultanément
- ✅ **Reprise incrémentale** : Reprend où ça s'est arrêté en cas d'interruption
- ✅ **Monitoring en temps réel** : Suivi de la progression globale
- ✅ **Gestion d'erreurs** : Isolation des erreurs par dataset
- ✅ **Upload automatique** : Upload direct sur HuggingFace

## Configuration

Le nombre de processus parallèles est configurable (recommandé: 3-4 max pour éviter la surcharge).

## Datasets supportés

1. CNN/DailyMail (deepsynth-en-news) - Priorité 1
2. arXiv (deepsynth-en-arxiv) - Priorité 2  
3. XSum BBC (deepsynth-en-xsum) - Priorité 3
4. MLSUM Français (deepsynth-fr) - Priorité 4
5. MLSUM Espagnol (deepsynth-es) - Priorité 5
6. MLSUM Allemand (deepsynth-de) - Priorité 6
7. BillSum Legal (deepsynth-en-legal) - Priorité 7