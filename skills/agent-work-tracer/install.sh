#!/bin/bash

# Agent Work Tracer - Installation Script
# Installs the skill to ~/.claude/skills/ for Claude to use

set -e

echo "🚀 Installing Agent Work Tracer Skill..."

# Create Claude skills directory if it doesn't exist
CLAUDE_SKILLS_DIR="$HOME/.claude/skills"
SKILL_NAME="agent-work-tracer"
SKILL_DIR="$CLAUDE_SKILLS_DIR/$SKILL_NAME"

echo "📁 Creating Claude skills directory..."
mkdir -p "$CLAUDE_SKILLS_DIR"

# Copy skill files
echo "📋 Copying skill files..."
if [ -d "$SKILL_DIR" ]; then
    echo "⚠️  Skill already exists. Updating..."
    rm -rf "$SKILL_DIR"
fi

cp -r "$(dirname "$0")" "$SKILL_DIR"

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x "$SKILL_DIR"/*.py
chmod +x "$SKILL_DIR/install.sh"

# Test the skill
echo "🧪 Testing skill installation..."
cd "$SKILL_DIR"
python3 trace_session.py --help > /dev/null
python3 handoff_generator.py --help > /dev/null

echo "✅ Agent Work Tracer skill installed successfully!"
echo ""
echo "📍 Installed to: $SKILL_DIR"
echo ""
echo "🎯 Usage:"
echo "  The skill will activate automatically when relevant."
echo "  You can also run manually:"
echo "    python3 $SKILL_DIR/trace_session.py"
echo "    python3 $SKILL_DIR/handoff_generator.py"
echo ""
echo "📚 Files created by the skill:"
echo "  - .agent_log.md (detailed work log)"
echo "  - .agent_status.json (machine-readable status)"
echo "  - .agent_handoff.md (quick handoff notes)"
echo ""
echo "🔄 The skill enables seamless agent-to-agent handoffs!"