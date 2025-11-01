#!/bin/bash
# Script de diagnostic pour le Mac Studio

echo "════════════════════════════════════════════════════════════════════"
echo "🔍 DIAGNOSTIC MAC STUDIO"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# 1. Processus
echo "1️⃣  PROCESSUS PYTHON:"
PROCS=$(ps aux | grep -E "python.*generate_qa" | grep -v grep)
if [ -z "$PROCS" ]; then
    echo "   ❌ AUCUN processus generate_qa_dataset ne tourne!"
    echo ""
    echo "   👉 Relancer avec:"
    echo "      cd $(pwd)"
    echo "      export PYTHONPATH=./src"
    echo "      python3 generate_qa_dataset.py 2>&1 | tee production.log &"
else
    echo "   ✅ Processus actif:"
    echo "$PROCS" | sed 's/^/      /'
fi
echo ""

# 2. Fichiers locaux
echo "2️⃣  FICHIERS LOCAUX:"
if [ -d "work/samples" ]; then
    BATCH_COUNT=$(ls work/samples/*.pkl 2>/dev/null | wc -l | tr -d ' ')
    if [ "$BATCH_COUNT" -gt 0 ]; then
        echo "   📁 Batches: $BATCH_COUNT"
        TOTAL_SIZE=$(du -sh work/samples 2>/dev/null | cut -f1)
        echo "   💾 Taille: $TOTAL_SIZE"

        # Compter échantillons (estimation)
        SAMPLE_EST=$((BATCH_COUNT * 50))
        echo "   📊 Échantillons estimés: ~$SAMPLE_EST"
    else
        echo "   ⚠️  Dossier work/samples/ vide"
    fi
else
    echo "   ❌ Dossier work/samples/ n'existe pas"
fi
echo ""

# 3. Logs récents
echo "3️⃣  LOGS RÉCENTS:"
if ls *.log 2>/dev/null | head -1 > /dev/null; then
    LATEST=$(ls -t *.log 2>/dev/null | head -1)
    echo "   📄 Dernier log: $LATEST"
    echo ""
    echo "   📋 Dernières 10 lignes:"
    tail -10 "$LATEST" 2>/dev/null | sed 's/^/      /'
    echo ""
    echo "   ⚠️  Erreurs récentes:"
    grep -i "error\|exception\|failed\|traceback" "$LATEST" 2>/dev/null | tail -5 | sed 's/^/      /' || echo "      Aucune erreur trouvée"
else
    echo "   ❌ Aucun fichier .log trouvé"
fi
echo ""

# 4. Version du code
echo "4️⃣  VERSION GIT:"
git log -1 --oneline 2>/dev/null || echo "   ⚠️  Pas dans un repo git"
echo ""
GIT_STATUS=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
if [ "$GIT_STATUS" -gt 0 ]; then
    echo "   ⚠️  $GIT_STATUS fichier(s) modifié(s) localement"
fi
echo ""

# 5. Config Python
echo "5️⃣  ENVIRONNEMENT PYTHON:"
if [ -f ".env" ]; then
    echo "   ✅ Fichier .env présent"
    if grep -q "HF_TOKEN" .env 2>/dev/null; then
        echo "   ✅ HF_TOKEN configuré"
    else
        echo "   ❌ HF_TOKEN manquant dans .env"
    fi
else
    echo "   ❌ Fichier .env absent"
fi
echo ""

# 6. État HuggingFace
echo "6️⃣  ÉTAT HUGGINGFACE:"
export PYTHONPATH=./src
python3 -c "
from huggingface_hub import HfApi
try:
    api = HfApi()
    info = api.repo_info('baconnier/deepsynth-qa', repo_type='dataset')
    print(f'   📅 Dernière MAJ HF: {info.last_modified}')
    files = list(api.list_repo_files('baconnier/deepsynth-qa', repo_type='dataset'))
    parquet = [f for f in files if '.parquet' in f]
    print(f'   📊 Fichiers Parquet: {len(parquet)}')
except Exception as e:
    print(f'   ❌ Erreur: {e}')
" 2>/dev/null || echo "   ❌ Impossible de vérifier HuggingFace"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "💡 ACTIONS RECOMMANDÉES"
echo "════════════════════════════════════════════════════════════════════"

if [ -z "$PROCS" ]; then
    echo ""
    echo "🚨 LE PROCESSUS NE TOURNE PAS!"
    echo ""
    echo "Étapes pour relancer:"
    echo "  1. Récupérer dernière version:"
    echo "     git pull origin main"
    echo ""
    echo "  2. Lancer la génération:"
    echo "     export PYTHONPATH=./src"
    echo "     python3 generate_qa_dataset.py 2>&1 | tee production.log &"
    echo ""
    echo "  3. Vérifier dans 5 minutes:"
    echo "     ./diagnostic_mac_studio.sh"
else
    echo ""
    echo "✅ Processus actif - surveillez les logs:"
    echo "   tail -f $LATEST"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
