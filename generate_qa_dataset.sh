#!/bin/bash
#
# Script de génération one-shot pour le dataset Q&A combiné
# Usage: ./generate_qa_dataset.sh [OPTIONS]
#
# Ce script va créer UN SEUL dataset combiné sur HuggingFace:
#   deepsynth-qa (~1.3M samples total)
#     ├─ Natural Questions (~300k)
#     └─ MS MARCO (~1M)
#
# Images pré-générées à résolution gundam (1600px)
# Source tracking: metadata.source = "natural_questions" | "ms_marco"
#
# Durée estimée: 6-12 heures (traitement séquentiel)
# Taille finale: 312-624 GB compressé sur HuggingFace
#

set -e  # Arrêt en cas d'erreur

echo "🔍 DEEPSYNTH - Génération Dataset Q&A Combiné"
echo "============================================================="

# Parse arguments
TEST_MODE=false
MAX_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --max-samples)
            MAX_SAMPLES="--max-samples $2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test              Test mode (1000 samples per source)"
            echo "  --max-samples N     Custom max samples for test mode"
            echo "  --help, -h          Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                         # Full generation (~1.3M samples)"
            echo "  $0 --test                  # Test with 1000 samples per source"
            echo "  $0 --max-samples 5000      # Test with 5000 samples per source"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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

echo ""

# Mode test ou production
if [ "$TEST_MODE" = true ]; then
    echo "🧪 MODE TEST"
    echo "============================================================="
    echo "📊 Traitement: ~2,000 samples (1,000 par source)"
    echo "⏱️  Temps estimé: 5-10 minutes"
    echo "💾 Taille finale: ~500 MB"
    echo ""
    TEST_FLAG="--test $MAX_SAMPLES"
else
    echo "🚀 MODE PRODUCTION"
    echo "============================================================="
    echo "📊 Traitement: ~1.3M samples total"
    echo "   • Natural Questions: ~300k samples"
    echo "   • MS MARCO: ~1M samples"
    echo ""
    echo "⏱️  Temps estimé: 6-12 heures"
    echo "💾 Taille finale: 312-624 GB compressé"
    echo ""
    echo "💡 NOTES:"
    echo "  • Images pré-générées à résolution gundam (1600px)"
    echo "  • Extraction contextuelle intelligente (Natural Questions)"
    echo "  • Upload incrémental tous les 5000 samples"
    echo "  • Interruption/reprise supportée (Ctrl+C)"
    echo ""
    echo "⚠️  AVERTISSEMENT: Ceci va prendre plusieurs heures"

    read -p "   Continuer? (o/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[OoYy]$ ]]; then
        echo "❌ Annulé par l'utilisateur"
        exit 0
    fi

    TEST_FLAG=""
fi

echo ""
echo "🚀 DÉMARRAGE DE LA GÉNÉRATION..."
echo "============================================================="
echo ""

# Lancer le script Python
chmod +x generate_qa_dataset.py
PYTHONPATH=./src python3 generate_qa_dataset.py $TEST_FLAG

EXIT_CODE=$?

echo ""
echo "============================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ GÉNÉRATION TERMINÉE AVEC SUCCÈS!"
    echo ""
    echo "📊 Dataset disponible sur:"
    echo "   https://huggingface.co/datasets/$(grep HF_USERNAME= .env | cut -d= -f2)/deepsynth-qa"
    echo ""
    echo "💡 Pour utiliser le dataset:"
    echo "   from datasets import load_dataset"
    echo "   dataset = load_dataset('$(grep HF_USERNAME= .env | cut -d= -f2)/deepsynth-qa')"
    echo ""
    echo "   # Filtrer par source"
    echo "   nq = dataset.filter(lambda x: x['metadata']['source'] == 'natural_questions')"
    echo "   marco = dataset.filter(lambda x: x['metadata']['source'] == 'ms_marco')"
else
    echo "⚠️  La génération s'est terminée avec des erreurs (code: $EXIT_CODE)"
    echo "💡 Consultez 'generate_qa_dataset.log' pour les détails"
fi
echo "============================================================="

exit $EXIT_CODE
