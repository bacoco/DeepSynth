# DeepSynth - Makefile pour commandes rapides
# Usage: make <command>

.PHONY: help setup test benchmark pipeline clean validate migrate

help: ## Afficher cette aide
	@echo "DeepSynth - Commandes disponibles:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Installer les dépendances
	@bash setup.sh

test: ## Lancer tous les tests
	@PYTHONPATH=./src python3 -m pytest tests/ -v

test-quick: ## Tests rapides (sans intégration)
	@PYTHONPATH=./src python3 -m pytest tests/ -v -m "not integration"

test-coverage: ## Tests avec couverture de code
	@PYTHONPATH=./src python3 -m pytest tests/ --cov=src/deepsynth --cov-report=html --cov-report=term

test-trainer: ## Tests du nouveau trainer optimisé
	@PYTHONPATH=./src python3 -m pytest tests/training/test_optimized_trainer.py -v

test-utils: ## Tests des utilitaires
	@PYTHONPATH=./src python3 -m pytest tests/utils/ -v

validate: ## Valider la qualité du codebase
	@python3 validate_codebase.py

fix-critical: ## Corriger les problèmes critiques
	@python3 fix_critical_issues.py

migrate: ## Analyser et générer guide de migration
	@python3 scripts/migrate_to_optimized_trainer.py --recursive src/

benchmark: ## Lancer les benchmarks métier
	@python scripts/cli/run_benchmark.py

benchmark-trainer: ## Benchmark performance du trainer
	@python3 scripts/benchmark_trainer_performance.py

pipeline: ## Lancer le pipeline complet
	@python scripts/cli/run_complete_multilingual_pipeline.py

pipeline-parallel: ## Pipeline avec traitement parallèle
	@python scripts/cli/run_parallel_processing.py

pipeline-global: ## Pipeline global cross-machine
	@bash scripts/run_global_pipeline.sh

example-trainer: ## Exemples d'utilisation du nouveau trainer
	@PYTHONPATH=./src python3 examples/train_with_optimized_trainer.py --all

clean: ## Nettoyer les fichiers temporaires
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache htmlcov .coverage 2>/dev/null || true
	@echo "✨ Nettoyage terminé"

web: ## Lancer l'interface web
	@python -m src.apps.web

format: ## Formater le code
	@black src/ tests/ scripts/ examples/ 2>/dev/null || echo "Installer black: pip install black"

lint: ## Vérifier le code
	@pylint src/ 2>/dev/null || echo "Installer pylint: pip install pylint"

quality-check: ## Vérification qualité complète (format + lint + validate)
	@echo "🔍 Vérification qualité du code..."
	@make format
	@make lint
	@make validate
	@echo "✅ Vérification terminée!"

# Cloud Workflows
generate-dataset: ## Générer et uploader un dataset vers HuggingFace
	@echo "Usage: make generate-dataset DATASET=mlsum_fr [MAX_SAMPLES=1000]"
	@python3 scripts/dataset_generation/generate_dataset_cloud.py $(DATASET) $(if $(MAX_SAMPLES),--max-samples $(MAX_SAMPLES))

generate-all-datasets: ## Générer TOUS les datasets
	@python3 scripts/dataset_generation/generate_dataset_cloud.py all

list-datasets: ## Lister les datasets générés sur HuggingFace
	@python3 scripts/dataset_generation/generate_dataset_cloud.py mlsum_fr --list

train-cloud: ## Entraîner depuis les datasets cloud
	@echo "Usage: make train-cloud DATASETS='mlsum_fr mlsum_es'"
	@python3 scripts/training/train_from_cloud_datasets.py --datasets $(or $(DATASETS),mlsum_fr)

train-cloud-all: ## Entraîner avec TOUS les datasets cloud
	@python3 scripts/training/train_from_cloud_datasets.py --datasets all --epochs 3
