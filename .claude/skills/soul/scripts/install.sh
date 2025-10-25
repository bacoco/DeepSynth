#!/bin/bash

# SOUL - Universal AI Agent Memory System
# Installs SOUL for any LLM (Claude, GPT, Gemini, LLaMA, etc.)

set -e

echo "🔮 Installing SOUL - Universal AI Agent Memory System..."
echo "   S.eamless O.rganized U.niversal L.earning"
echo ""

# Parse command line arguments
MODEL="claude"  # Default to Claude
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--model claude|gpt|gemini|universal]"
            exit 1
            ;;
    esac
done

echo "🎯 Installing SOUL for: $MODEL"
echo ""

# Universal installation
SOUL_DIR="$HOME/.soul"
echo "📁 Creating SOUL directory..."
mkdir -p "$SOUL_DIR"

# Copy SOUL files
echo "📋 Copying SOUL files..."
if [ -d "$SOUL_DIR" ]; then
    echo "⚠️  SOUL already exists. Updating..."
    rm -rf "$SOUL_DIR"/*
fi

cp -r "$(dirname "$0")"/* "$SOUL_DIR/"

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x "$SOUL_DIR"/*.py
chmod +x "$SOUL_DIR"/*.sh

# Model-specific installation
case $MODEL in
    claude)
        echo "🤖 Installing for Claude Skills..."
        CLAUDE_SKILLS_DIR="$HOME/.claude/skills"
        mkdir -p "$CLAUDE_SKILLS_DIR"
        ln -sf "$SOUL_DIR" "$CLAUDE_SKILLS_DIR/soul"
        echo "✅ Claude Skills integration complete"
        ;;
    gpt)
        echo "🧠 Installing for GPT..."
        cat > "$HOME/.soul_gpt_instructions.txt" << 'EOF'
You are an AI agent with SOUL (Seamless Organized Universal Learning).

IMPORTANT: At the start of each session, read:
- .agent_log.md (complete work history)
- .agent_handoff.md (immediate next steps)  
- .agent_status.json (machine-readable state)

At the end of each session, run:
python3 ~/.soul/trace_session.py --verbose

This gives you persistent memory across sessions.
EOF
        echo "✅ GPT instructions saved to ~/.soul_gpt_instructions.txt"
        ;;
    gemini)
        echo "💎 Installing for Gemini..."
        cat > "$HOME/.soul_gemini_prompt.txt" << 'EOF'
You have access to SOUL - a universal AI memory system.

INITIALIZATION: Always start by reading these files:
1. .agent_log.md - Previous agents' work history
2. .agent_handoff.md - Priority actions for you
3. .agent_status.json - Current project state

TERMINATION: Before ending, always run:
python3 ~/.soul/trace_session.py --verbose
python3 ~/.soul/handoff_generator.py --both
EOF
        echo "✅ Gemini prompt saved to ~/.soul_gemini_prompt.txt"
        ;;
    universal)
        echo "🌍 Installing for Universal LLM usage..."
        cat > "$HOME/.soul_universal_guide.txt" << 'EOF'
SOUL - Universal AI Memory System

Add this to any LLM's system prompt:

"You are equipped with SOUL memory system.
Before starting: read .agent_log.md for context.
Before ending: run python3 ~/.soul/trace_session.py"

For API usage, load .agent_handoff.md and include in context.
EOF
        echo "✅ Universal guide saved to ~/.soul_universal_guide.txt"
        ;;
esac

# Test the installation
echo "🧪 Testing SOUL installation..."
cd "$SOUL_DIR"
python3 trace_session.py --help > /dev/null
python3 handoff_generator.py --help > /dev/null

echo ""
echo "🎉 SOUL installed successfully!"
echo ""
echo "📍 SOUL Location: $SOUL_DIR"
echo "🎯 Model: $MODEL"
echo ""
echo "🔮 SOUL Features:"
echo "  ✅ Persistent memory across AI sessions"
echo "  ✅ Cross-model collaboration (Claude ↔ GPT ↔ Gemini)"
echo "  ✅ Universal problem-solution database"
echo "  ✅ Seamless handoffs between different AIs"
echo ""
echo "📚 Files created by SOUL:"
echo "  - .agent_log.md (detailed work history)"
echo "  - .agent_status.json (machine-readable status)"
echo "  - .agent_handoff.md (immediate next steps)"
echo ""
echo "🚀 Your AI agents now have a SOUL!"
echo "   They can remember, learn, and collaborate across sessions."
echo ""
echo "💡 Next steps:"
case $MODEL in
    claude)
        echo "  - SOUL will activate automatically in Claude"
        echo "  - No additional setup needed"
        ;;
    gpt)
        echo "  - Add contents of ~/.soul_gpt_instructions.txt to GPT custom instructions"
        echo "  - SOUL will provide persistent memory"
        ;;
    gemini)
        echo "  - Use ~/.soul_gemini_prompt.txt as system prompt"
        echo "  - SOUL will enable cross-session continuity"
        ;;
    universal)
        echo "  - See ~/.soul_universal_guide.txt for integration with any LLM"
        echo "  - SOUL works with Claude, GPT, Gemini, LLaMA, and more"
        ;;
esac