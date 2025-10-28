#!/bin/bash
# Script à lancer sur le Mac Studio pour monitorer la génération

echo "════════════════════════════════════════════════════════════════════"
echo "📊 MONITORING GÉNÉRATION Q&A DATASET"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# 1. Vérifier si le processus tourne
echo "🔍 PROCESSUS EN COURS:"
if ps aux | grep -E "generate_qa_dataset|PYTHONPATH.*deepsynth" | grep -v grep | grep -v monitor; then
    echo "   ✅ Processus actif"
else
    echo "   ⚠️  Aucun processus détecté!"
fi
echo ""

# 2. Progression locale
echo "💾 FICHIERS LOCAUX:"
if [ -d "qa_samples" ]; then
    BATCH_COUNT=$(ls qa_samples/*.jsonl 2>/dev/null | wc -l)
    TOTAL_SIZE=$(du -sh qa_samples 2>/dev/null | cut -f1)
    echo "   📁 Batches: $BATCH_COUNT"
    echo "   💿 Taille: $TOTAL_SIZE"

    # Compter les échantillons dans les batches
    if [ $BATCH_COUNT -gt 0 ]; then
        SAMPLE_COUNT=$(cat qa_samples/*.jsonl 2>/dev/null | wc -l)
        echo "   📊 Échantillons générés: $SAMPLE_COUNT"
        echo "   🎯 Prochain upload à: 5000 échantillons"

        REMAINING=$((5000 - SAMPLE_COUNT))
        if [ $REMAINING -gt 0 ]; then
            echo "   ⏳ Reste: $REMAINING échantillons"
        else
            echo "   🚀 Upload imminent!"
        fi
    fi
else
    echo "   ⚠️  Dossier qa_samples/ absent"
fi
echo ""

# 3. Logs récents
echo "📋 DERNIERS LOGS (5 minutes):"
if ls -t *.log 2>/dev/null | head -1 > /dev/null; then
    LATEST_LOG=$(ls -t *.log 2>/dev/null | head -1)
    echo "   📄 Fichier: $LATEST_LOG"
    echo "   🕐 Dernière ligne:"
    tail -1 "$LATEST_LOG" 2>/dev/null | sed 's/^/      /'
    echo ""
    echo "   📈 Progression (dernières 10 lignes):"
    tail -10 "$LATEST_LOG" 2>/dev/null | grep -E "Converted|samples|batch|Upload" | sed 's/^/      /'
else
    echo "   ⚠️  Aucun fichier .log trouvé"
fi
echo ""

# 4. État HuggingFace
echo "☁️  ÉTAT HUGGINGFACE:"
PYTHONPATH=./src python3 -c "
from huggingface_hub import HfApi
from datetime import datetime
try:
    api = HfApi()
    info = api.repo_info('baconnier/deepsynth-qa', repo_type='dataset')
    last_update = info.last_modified.strftime('%Y-%m-%d %H:%M:%S')
    print(f'   📅 Dernière MAJ: {last_update}')

    files = list(api.list_repo_files('baconnier/deepsynth-qa', repo_type='dataset'))
    parquet = [f for f in files if f.endswith('.parquet')]
    print(f'   📊 Fichiers Parquet: {len(parquet)}')
except Exception as e:
    print(f'   ❌ Erreur: {e}')
" 2>/dev/null
echo ""

# 5. Estimation temps
echo "⏱️  ESTIMATION:"
if [ -d "qa_samples" ] && [ $BATCH_COUNT -gt 0 ]; then
    # Estimation grossière: 3-5 sec par échantillon
    SAMPLE_COUNT=$(cat qa_samples/*.jsonl 2>/dev/null | wc -l)
    if [ $SAMPLE_COUNT -gt 10 ]; then
        REMAINING=$((5000 - SAMPLE_COUNT))
        MIN_TIME=$((REMAINING * 3 / 60))
        MAX_TIME=$((REMAINING * 5 / 60))
        echo "   ⏳ Temps estimé avant upload: ${MIN_TIME}-${MAX_TIME} minutes"
        echo "   🎯 Cible MS MARCO: ~140,000 échantillons"
        echo "   🎯 Cible Natural Questions: ~300,000 échantillons"
    fi
fi
echo ""

echo "════════════════════════════════════════════════════════════════════"
echo "💡 Commandes utiles:"
echo "   tail -f *.log                    # Suivre les logs en temps réel"
echo "   ./monitor_remote.sh              # Relancer ce monitoring"
echo "   ps aux | grep generate_qa        # Vérifier le processus"
echo "════════════════════════════════════════════════════════════════════"
