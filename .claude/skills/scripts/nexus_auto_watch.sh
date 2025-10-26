#!/bin/bash
# NEXUS Auto Watch - Surveille et génère automatiquement les skills
# Ce script peut être exécuté:
# - Manuellement: ./nexus_auto_watch.sh
# - Via cron: */30 * * * * /path/to/nexus_auto_watch.sh
# - Via git hook: dans .git/hooks/post-commit

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEXUS_DIR="$SCRIPT_DIR/../nexus/scripts"
LOG_FILE="/tmp/nexus_auto_watch.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') - NEXUS Auto Watch started" >> "$LOG_FILE"

# Exécuter le générateur automatique
cd "$SCRIPT_DIR/../../.."

if python3 "$NEXUS_DIR/auto_skill_generator.py" >> "$LOG_FILE" 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Auto-generation completed" >> "$LOG_FILE"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Auto-generation failed" >> "$LOG_FILE"
fi
