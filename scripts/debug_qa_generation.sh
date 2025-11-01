#!/bin/bash
# Script de diagnostic pour debug la génération Q&A

echo "============================================================"
echo "🔍 DIAGNOSTIC GÉNÉRATION Q&A"
echo "============================================================"
echo ""

# 1. Vérifier si le processus tourne
echo "1️⃣ PROCESSUS EN COURS:"
ps aux | grep generate_qa_dataset | grep -v grep | grep -v debug
if [ $? -ne 0 ]; then
    echo "   ❌ Aucun processus generate_qa_dataset détecté"
else
    echo "   ✅ Processus actif"
fi
echo ""

# 2. Vérifier les logs
echo "2️⃣ LOGS RÉCENTS (dernières 50 lignes):"
if [ -f "generate_qa_dataset.log" ]; then
    echo "   📄 generate_qa_dataset.log:"
    tail -50 generate_qa_dataset.log
else
    echo "   ⚠️  Pas de fichier generate_qa_dataset.log"
fi
echo ""

# 3. Vérifier les fichiers temporaires
echo "3️⃣ FICHIERS TEMPORAIRES:"
if [ -d "work" ]; then
    echo "   📁 work/:"
    ls -lh work/ 2>/dev/null
    if [ -d "work/samples" ]; then
        echo "   📁 work/samples/:"
        ls -lh work/samples/ 2>/dev/null | head -10
    fi
else
    echo "   ⚠️  Pas de dossier work/"
fi
echo ""

# 4. Vérifier l'utilisation CPU/Mémoire
echo "4️⃣ UTILISATION RESSOURCES:"
ps aux | grep python | grep -v grep | head -5
echo ""

# 5. Vérifier la connexion réseau
echo "5️⃣ CONNEXION HUGGINGFACE:"
curl -s -I https://huggingface.co | head -1
echo ""

# 6. Vérifier les variables d'environnement
echo "6️⃣ CONFIGURATION:"
if [ -f ".env" ]; then
    echo "   ✅ Fichier .env présent"
    grep "HF_USERNAME" .env || echo "   ⚠️  HF_USERNAME manquant"
    if grep -q "HF_TOKEN=.\+" .env; then
        echo "   ✅ HF_TOKEN configuré"
    else
        echo "   ❌ HF_TOKEN manquant ou vide"
    fi
else
    echo "   ❌ Pas de fichier .env"
fi
echo ""

# 7. Tester l'import Python
echo "7️⃣ TEST IMPORTS PYTHON:"
PYTHONPATH=./src python3 -c "
from deepsynth.data.dataset_converters import convert_natural_questions
print('✅ Import converter OK')
" 2>&1
echo ""

echo "============================================================"
echo "💡 ACTIONS SUGGÉRÉES:"
echo ""
echo "Si aucun processus actif:"
echo "  → Relancer: ./generate_qa_dataset.sh"
echo ""
echo "Si processus bloqué:"
echo "  → Tuer: pkill -f generate_qa_dataset"
echo "  → Relancer: ./generate_qa_dataset.sh"
echo ""
echo "Si problème réseau:"
echo "  → Vérifier connexion internet"
echo "  → Vérifier HF_TOKEN dans .env"
echo ""
echo "Pour voir les logs en temps réel:"
echo "  → tail -f generate_qa_dataset.log"
echo "============================================================"
