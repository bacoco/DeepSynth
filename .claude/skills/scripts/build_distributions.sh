#!/bin/bash
# Build distribution zips for all skills
# Generates standalone packages for SOUL, NEXUS, and skill-generator

set -e

SKILLS_DIR=".claude/skills"
OUTPUT_DIR=".claude/skills"

echo "🔧 Building skill distribution zips..."
echo ""

# Array of skills to package
SKILLS=("soul" "nexus" "skill-creator")

for skill in "${SKILLS[@]}"; do
    echo "📦 Packaging ${skill}..."

    cd "$SKILLS_DIR/$skill"

    # Create zip excluding unnecessary files
    zip -r "../${skill}.zip" . \
        -x "*.pyc" \
        -x "__pycache__/*" \
        -x "*.zip" \
        -x ".DS_Store" \
        -x "*.swp" \
        -x "*.bak" \
        > /dev/null 2>&1

    cd - > /dev/null

    # Get zip size
    SIZE=$(du -h "$SKILLS_DIR/${skill}.zip" | cut -f1)
    echo "   ✓ ${skill}.zip created (${SIZE})"
    echo ""
done

echo "✅ All distribution zips created in ${OUTPUT_DIR}/"
echo ""
echo "Available packages:"
ls -lh "$OUTPUT_DIR"/*.zip | awk '{print "  -", $9, "("$5")"}'
echo ""
echo "Usage:"
echo "  unzip soul.zip"
echo "  cd soul && ./scripts/install.sh --model=claude"
