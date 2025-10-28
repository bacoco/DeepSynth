#!/bin/bash

# Script de génération complète du dataset Q&A
# Génère MS MARCO (~1M samples) + Natural Questions (~307k samples)
# Durée estimée: 3-5 heures

echo "🚀 Démarrage génération dataset Q&A complet..."
echo "📊 Cible: baconnier/deepsynth-qa sur HuggingFace"
echo "⏱️  Durée estimée: 3-5 heures"
echo ""

# Configuration
export PYTHONPATH=./src

# Lancement
python3 generate_qa_dataset.py 2>&1 | tee qa_full_generation.log

echo ""
echo "✅ Génération terminée !"
echo "📝 Logs sauvegardés dans: qa_full_generation.log"
echo "🔗 Dataset: https://huggingface.co/datasets/baconnier/deepsynth-qa"
