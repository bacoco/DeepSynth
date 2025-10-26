#!/bin/bash
#
# Script de mise à jour du repo DeepSynth
# Usage: ./update.sh
#
# Ce script nettoie les fichiers temporaires et fait un git pull propre
#

set -e

echo "🔄 MISE À JOUR DU REPO DEEPSYNTH"
echo "================================="

# Nettoyer les fichiers macOS
echo "🧹 Nettoyage des fichiers système..."
find . -name ".DS_Store" -type f -delete 2>/dev/null || true
find . -name "*.pyc" -type f -delete 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Nettoyer les fichiers de travail temporaires
echo "🧹 Nettoyage des fichiers temporaires de travail..."
rm -rf work_separate_* 2>/dev/null || true
rm -f parallel_datasets.log 2>/dev/null || true

# Sauvegarder .env si existe
if [ -f ".env" ]; then
    echo "💾 Sauvegarde de .env..."
    cp .env .env.backup
fi

# Stash les changements locaux (si besoin)
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo "📦 Sauvegarde des changements locaux..."
    git stash push -m "Auto-stash before update $(date)"
    STASHED=1
else
    STASHED=0
fi

# Pull les changements
echo "⬇️  Récupération des mises à jour..."
git pull --rebase origin main

# Restaurer le stash si nécessaire
if [ $STASHED -eq 1 ]; then
    echo "📦 Restauration des changements locaux..."
    git stash pop || {
        echo "⚠️  Conflit lors de la restauration du stash"
        echo "   Résolvez manuellement avec: git stash list && git stash pop"
    }
fi

# Restaurer .env
if [ -f ".env.backup" ]; then
    echo "💾 Restauration de .env..."
    mv .env.backup .env
fi

# Mettre à jour les dépendances Python si venv existe
if [ -d "venv" ]; then
    echo "📦 Mise à jour des dépendances Python..."
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo "✅ Dépendances mises à jour"
else
    echo "ℹ️  Pas de venv détecté - Lancez ./setup.sh pour l'installer"
fi

echo ""
echo "================================="
echo "✅ MISE À JOUR TERMINÉE"
echo "================================="
echo ""
echo "💡 Si vous aviez des changements locaux, vérifiez:"
echo "   git status"
echo "   git stash list"
