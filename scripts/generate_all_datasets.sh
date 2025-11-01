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
# Chaque dataset contiendra des images originales haute qualité
#   Augmentation aléatoire appliquée pendant l'entraînement:
#   - Rotation (±10°), perspective, resize (512-1600px), color jitter
#
# Durée estimée: 2-4 heures (6x plus rapide qu'avant!)
# Espace disque requis: ~2.5GB temporaire (6x moins qu'avant!)
#

set -e  # Arrêt en cas d'erreur

echo "🌍 DEEPSYNTH - Génération de tous les datasets en parallèle"
echo "============================================================="

# Vérifier et installer les dépendances si nécessaire
if [ ! -d "venv" ] || [ ! -f "/Library/Fonts/DejaVuSans.ttf" ]; then
    echo "⚙️  Installation des dépendances et fonts Unicode..."
    echo "   (Première exécution uniquement - peut demander sudo)"
    echo ""

    if [ ! -x "./setup.sh" ]; then
        chmod +x setup.sh
    fi

    ./setup.sh

    echo ""
    echo "✅ Installation terminée"
    echo "============================================================="
    echo ""
fi

# Activer l'environnement virtuel
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Environnement virtuel activé"
else
    echo "⚠️  Pas d'environnement virtuel détecté"
    echo "   Utilisation de Python système"
fi

echo ""

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
echo "⏱️  Temps estimé: 2-4 heures (6x plus rapide!)"
echo "🔄 Traitement parallèle: 7 workers (1 par dataset)"
echo "📤 Upload automatique tous les 5000 samples"
echo "💾 Espace économisé: 6x moins de stockage (images originales uniquement)"
echo ""
echo "💡 NOTES:"
echo "  • Vous pouvez interrompre (Ctrl+C) et reprendre plus tard"
echo "  • Les datasets seront visibles sur HuggingFace au fur et à mesure"
echo "  • Les logs détaillés sont dans 'parallel_datasets.log'"
echo ""

# Nettoyer les anciens work directories de test
echo "🧹 Nettoyage des anciens work directories de test..."
if ls work_separate* work_* 2>/dev/null | grep -q .; then
    rm -rf work_separate* work_* 2>/dev/null || true
    echo "✅ Anciens work directories supprimés"
else
    echo "✅ Aucun ancien work directory à nettoyer"
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
