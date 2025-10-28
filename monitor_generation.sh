#!/bin/bash
# Moniteur temps réel de la génération Q&A

echo "🔍 MONITEUR GÉNÉRATION Q&A - Rafraîchissement toutes les 5 secondes"
echo "================================================================="
echo ""

while true; do
    clear
    echo "🕐 $(date '+%H:%M:%S')"
    echo "================================================================="

    # Processus Python actif
    echo ""
    echo "1️⃣ PROCESSUS PYTHON:"
    ps aux | grep "[p]ython.*generate_qa" | awk '{printf "   PID: %s | CPU: %s%% | MEM: %s%% | Temps: %s\n", $2, $3, $4, $10}' || echo "   ❌ Aucun processus"

    # Activité réseau (téléchargement HuggingFace)
    echo ""
    echo "2️⃣ ACTIVITÉ RÉSEAU (datasets HuggingFace):"
    lsof -i -n | grep -i python | grep ESTABLISHED | head -5 || echo "   Aucune connexion active"

    # Fichiers générés
    echo ""
    echo "3️⃣ FICHIERS GÉNÉRÉS:"
    if [ -d "work/samples" ]; then
        BATCH_COUNT=$(ls -1 work/samples/*.pkl 2>/dev/null | wc -l | tr -d ' ')
        if [ "$BATCH_COUNT" -gt 0 ]; then
            echo "   📦 Batches: $BATCH_COUNT fichiers"
            ls -lh work/samples/*.pkl 2>/dev/null | tail -5
        else
            echo "   ⏳ Pas encore de batches (téléchargement en cours...)"
        fi
    else
        echo "   ⏳ Dossier work/samples non créé"
    fi

    # Dernières lignes du log
    echo ""
    echo "4️⃣ DERNIERS LOGS (5 dernières lignes):"
    if [ -f "generate_qa_dataset.log" ]; then
        tail -5 generate_qa_dataset.log | sed 's/^/   /'
    else
        echo "   ⏳ Pas encore de fichier log"
    fi

    # Espace disque
    echo ""
    echo "5️⃣ ESPACE DISQUE:"
    df -h . | tail -1 | awk '{printf "   Utilisé: %s / %s (%s)\n", $3, $2, $5}'

    echo ""
    echo "================================================================="
    echo "💡 Ctrl+C pour quitter le moniteur (le processus continuera)"
    echo "================================================================="

    sleep 5
done
