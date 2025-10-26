#!/bin/bash
#
# Script de lancement pour générer TOUS les datasets en parallèle
# Usage: ./generate_all_datasets.sh
#
# Ce script va créer les 7 datasets multilingues sur HuggingFace:
#   1. deepsynth-en-news   (CNN/DailyMail ~287k)
#   2. deepsynth-en-arxiv  (arXiv Scientific ~50k)
#   3. deepsynth-en-xsum   (BBC XSum ~50k)
#   4. deepsynth-fr        (MLSUM Français ~392k)
#   5. deepsynth-es        (MLSUM Espagnol ~266k)
#   6. deepsynth-de        (MLSUM Allemand ~220k)
#   7. deepsynth-en-legal  (BillSum Legal ~22k)
#
# Durée estimée: 6-12 heures
# Espace disque requis: ~15GB temporaire
#

set -e  # Arrêt en cas d'erreur

echo "🌍 DEEPSYNTH - Génération de tous les datasets en parallèle"
echo "============================================================="

# Vérifier que le fichier .env existe
if [ ! -f .env ]; then
    echo "❌ Erreur: Fichier .env introuvable"
    echo "💡 Copiez .env.example vers .env et configurez HF_TOKEN"
    exit 1
fi

# Vérifier HF_TOKEN
if ! grep -q "HF_TOKEN=" .env; then
    echo "❌ Erreur: HF_TOKEN non configuré dans .env"
    echo "💡 Ajoutez votre token HuggingFace dans le fichier .env"
    exit 1
fi

# Afficher la configuration
echo "✅ Configuration détectée:"
grep "HF_USERNAME=" .env || echo "⚠️  HF_USERNAME non défini"
grep "ARXIV_IMAGE_SAMPLES=" .env || echo "ℹ️  ARXIV_IMAGE_SAMPLES: utilise défaut (50000)"

echo ""
echo "📊 Ce script va traiter ~1.29M échantillons"
echo "⏱️  Temps estimé: 6-12 heures"
echo "🔄 Traitement parallèle: 3 workers"
echo ""
echo "💡 NOTES:"
echo "  • Vous pouvez interrompre (Ctrl+C) et reprendre plus tard"
echo "  • Les datasets seront visibles sur HuggingFace au fur et à mesure"
echo "  • Les logs détaillés sont dans 'parallel_datasets.log'"
echo ""

# Demander confirmation
read -p "🚀 Démarrer le traitement? (o/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[OoYy]$ ]]; then
    echo "⏹️  Annulé par l'utilisateur"
    exit 0
fi

echo ""
echo "🚀 DÉMARRAGE DU TRAITEMENT..."
echo "============================================================="
echo ""

# Lancer le pipeline Python
PYTHONPATH=./src python3 run_full_pipeline.py

EXIT_CODE=$?

echo ""
echo "============================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TRAITEMENT TERMINÉ AVEC SUCCÈS!"
else
    echo "⚠️  Le traitement s'est terminé avec des erreurs (code: $EXIT_CODE)"
    echo "💡 Consultez 'parallel_datasets.log' pour les détails"
fi
echo "============================================================="

exit $EXIT_CODE
