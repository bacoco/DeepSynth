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

# ========================================
# Unsloth DeepSeek OCR Targets (NEW!)
# ========================================

deepseek-ocr-smoke: ## Quick smoke test with Unsloth (100 samples, 1 epoch)
	@echo "🧪 Running Unsloth smoke test..."
	@PYTHONPATH=./src python3 scripts/train_unsloth_cli.py \
		--max_train_samples 100 \
		--max_eval_samples 50 \
		--num_epochs 1 \
		--batch_size 2 \
		--output_dir ./output/smoke-test
	@echo "✅ Smoke test complete!"

deepseek-ocr-train: ## Full training with Unsloth optimizations
	@echo "🚀 Starting Unsloth training (1.4x faster, 40% less VRAM)..."
	@PYTHONPATH=./src python3 scripts/train_unsloth_cli.py \
		--dataset_name ccdv/cnn_dailymail \
		--batch_size 4 \
		--num_epochs 3 \
		--use_wandb \
		--push_to_hub \
		--output_dir ./output/deepsynth-unsloth
	@echo "✅ Training complete!"

deepseek-ocr-eval: ## Evaluate trained Unsloth model
	@echo "📊 Evaluating model..."
	@PYTHONPATH=./src python3 scripts/evaluate_ocr.py \
		--model_path ./output/deepsynth-unsloth/final_model \
		--dataset_name ccdv/cnn_dailymail \
		--split validation \
		--num_samples 1000
	@echo "✅ Evaluation complete!"

deepseek-ocr-benchmark: ## Benchmark Unsloth vs Standard trainer
	@echo "⚡ Benchmarking Unsloth vs Standard..."
	@PYTHONPATH=./src python3 scripts/benchmark_unsloth_vs_standard.py
	@echo "✅ Benchmark complete!"

deepseek-ocr-prepare-data: ## Prepare OCR dataset from WebDataset/Parquet
	@echo "📦 Preparing OCR dataset..."
	@echo "Usage: make deepseek-ocr-prepare-data SOURCE=<url> TYPE=<webdataset|parquet>"
	@PYTHONPATH=./src python3 scripts/training/prepare_ocr_dataset.py \
		--source $(SOURCE) \
		--source_type $(TYPE) \
		--convert_to_images \
		--output_path ./data/ocr-prepared

# Docker targets
docker-build-unsloth: ## Build Unsloth Docker image
	@echo "🐳 Building Unsloth Docker image..."
	@docker build -f docker/deepseek-ocr.Dockerfile -t deepsynth-unsloth:latest .
	@echo "✅ Docker image built: deepsynth-unsloth:latest"

docker-run-smoke: ## Run smoke test in Docker
	@echo "🐳 Running smoke test in Docker..."
	@docker run --gpus all -v $(PWD)/output:/output deepsynth-unsloth:latest \
		python scripts/train_unsloth_cli.py \
		--max_train_samples 100 \
		--num_epochs 1 \
		--batch_size 2 \
		--output_dir /output/smoke-test

docker-run-train: ## Run full training in Docker
	@echo "🐳 Running training in Docker..."
	@docker run --gpus all \
		-v $(PWD)/data:/data \
		-v $(PWD)/output:/output \
		-v $(PWD)/.cache:/workspace/.cache \
		-e HF_TOKEN=$(HF_TOKEN) \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		deepsynth-unsloth:latest \
		python scripts/train_unsloth_cli.py \
		--dataset_name ccdv/cnn_dailymail \
		--batch_size 4 \
		--num_epochs 3 \
		--use_wandb \
		--output_dir /output/deepsynth-unsloth

docker-compose-up: ## Start services with docker-compose
	@cd docker && docker-compose up

docker-compose-smoke: ## Run smoke test with docker-compose
	@cd docker && docker-compose run deepsynth-smoke

# Monitoring targets
monitor-tensorboard: ## Start TensorBoard for monitoring
	@echo "📊 Starting TensorBoard..."
	@tensorboard --logdir ./output/deepsynth-unsloth/logs --port 6006
	@echo "📊 TensorBoard available at http://localhost:6006"

monitor-prometheus: ## Export Prometheus metrics
	@echo "📊 Prometheus metrics available at http://localhost:9090/metrics"
	@curl http://localhost:9090/metrics 2>/dev/null || echo "API server not running. Start with: make api-server"

# API server targets
api-server: ## Start OCR inference API server
	@echo "🚀 Starting OCR API server..."
	@PYTHONPATH=./src python3 -m deepsynth.inference.api_server \
		--model_path ./output/deepsynth-unsloth/final_model \
		--port 8000 \
		--enable_metrics
	@echo "🚀 API available at http://localhost:8000"

api-test: ## Test OCR API endpoint
	@echo "🧪 Testing OCR API..."
	@curl -X POST http://localhost:8000/api/ocr/run \
		-F "file=@examples/sample_document.png" \
		|| echo "❌ API not running or sample file missing"

# Help for Unsloth targets
help-unsloth: ## Show Unsloth-specific help
	@echo "🚀 Unsloth DeepSeek OCR Commands:"
	@echo ""
	@echo "Training:"
	@echo "  make deepseek-ocr-smoke          - Quick 5min test (100 samples)"
	@echo "  make deepseek-ocr-train          - Full training (1.4x faster!)"
	@echo "  make deepseek-ocr-eval           - Evaluate model (CER/WER/ROUGE)"
	@echo "  make deepseek-ocr-benchmark      - Compare Unsloth vs Standard"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build-unsloth        - Build Docker image"
	@echo "  make docker-run-smoke            - Run smoke test in Docker"
	@echo "  make docker-run-train            - Run training in Docker"
	@echo "  make docker-compose-up           - Start all services"
	@echo ""
	@echo "Monitoring:"
	@echo "  make monitor-tensorboard         - View training metrics"
	@echo "  make api-server                  - Start inference API"
	@echo "  make api-test                    - Test API endpoint"
	@echo ""
	@echo "Expected Performance:"
	@echo "  - Training: 1.4x faster (12h → 8.5h)"
	@echo "  - VRAM: 40% reduction (24GB → 14GB)"
	@echo "  - Context: 5x longer (1024 → 5120 tokens)"
	@echo "  - CER: 88% improvement"
