# DeepSynth - Makefile pour commandes rapides
# Usage: make <command>

.PHONY: help setup test benchmark pipeline clean

help: ## Afficher cette aide
	@echo "DeepSynth - Commandes disponibles:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Installer les dépendances
	@bash setup.sh

test: ## Lancer tous les tests
	@python -m pytest tests/ -v

test-quick: ## Tests rapides (sans intégration)
	@python -m pytest tests/ -v -m "not integration"

benchmark: ## Lancer les benchmarks
	@python scripts/run_benchmark.py

pipeline: ## Lancer le pipeline complet
	@python scripts/run_complete_multilingual_pipeline.py

pipeline-parallel: ## Pipeline avec traitement parallèle
	@python scripts/run_parallel_processing.py

pipeline-global: ## Pipeline global cross-machine
	@bash scripts/run_global_pipeline.sh

clean: ## Nettoyer les fichiers temporaires
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✨ Nettoyage terminé"

web: ## Lancer l'interface web
	@python -m src.apps.web

format: ## Formater le code
	@black src/ tests/ scripts/ 2>/dev/null || echo "Installer black: pip install black"

lint: ## Vérifier le code
	@pylint src/ 2>/dev/null || echo "Installer pylint: pip install pylint"
