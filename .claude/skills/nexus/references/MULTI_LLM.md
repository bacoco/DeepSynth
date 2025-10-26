# NEXUS Multi-LLM Support

Using NEXUS with different LLM platforms.

## Universal Output Format

NEXUS uses LLM-agnostic formats:
- **Markdown** for human-readable recommendations
- **JSON** for machine-readable data
- **Python** for programmatic access

No LLM-specific syntax required.

## Supported LLMs

### Claude Code (Native)

**How It Works:**
- NEXUS runs automatically via cron/git hooks
- Auto-generates skills when patterns detected
- Skills immediately available to Claude
- Full integration with SOUL memory

**Setup:**

```bash
# Run manually
python .claude/skills/nexus/scripts/auto_skill_generator.py

# Or setup automatic monitoring
*/30 * * * * /path/to/.claude/skills/scripts/nexus_auto_watch.sh
```

**Usage:**

```bash
# Claude automatically uses generated skills
claude

# NEXUS has already created skills based on your patterns
# Claude loads them automatically
```

**Features:**
- ✅ Automatic skill generation
- ✅ SOUL integration
- ✅ Immediate skill loading
- ✅ Full progressive disclosure support

---

### OpenAI GPT (CLI)

**How It Works:**
- NEXUS runs in background
- Generates `NEXUS_RECOMMENDATIONS.md`
- GPT reads recommendations manually or via system prompt

**Setup:**

```bash
# 1. Setup NEXUS monitoring (same as Claude)
*/30 * * * * /path/to/.claude/skills/scripts/nexus_auto_watch.sh

# 2. Configure GPT to read recommendations
cat > ~/.config/gpt/system_prompt.txt << 'EOF'
Check NEXUS_RECOMMENDATIONS.md for skill recommendations.
If high-priority skills are recommended, notify the user.
EOF
```

**Usage:**

```bash
# Start GPT session
gpt

# Ask about recommendations
> Check NEXUS recommendations

# GPT reads NEXUS_RECOMMENDATIONS.md and reports:
"NEXUS recommends 2 high-priority skills:
1. api-optimizer (CRITICAL)
2. test-guardian (HIGH)

Would you like me to create these skills?"
```

**Manual Workflow:**

```bash
# 1. Run NEXUS analysis
python .claude/skills/nexus/scripts/nexus_analyzer.py

# 2. Read recommendations
cat NEXUS_RECOMMENDATIONS.md

# 3. Ask GPT to create recommended skills
gpt
> Create the api-optimizer skill as recommended by NEXUS
```

**Features:**
- ✅ Reads recommendations
- ⚠️ Manual skill creation
- ⚠️ No automatic monitoring
- ✅ SOUL integration (via Python API)

---

### Google Gemini (CLI)

**How It Works:**
- NEXUS generates recommendations
- Gemini reads markdown recommendations
- Manual or semi-automatic skill creation

**Setup:**

```bash
# 1. Setup NEXUS monitoring
*/30 * * * * /path/to/.claude/skills/scripts/nexus_auto_watch.sh

# 2. Configure Gemini startup script
cat > ~/.gemini/startup.sh << 'EOF'
#!/bin/bash
if [ -f "NEXUS_RECOMMENDATIONS.md" ]; then
    echo "📊 NEXUS recommendations available. Type 'check nexus' to review."
fi
EOF
```

**Usage:**

```bash
# Start Gemini
gemini

# Check recommendations
> Read NEXUS_RECOMMENDATIONS.md and summarize

# Create recommended skills
> Create the api-optimizer skill following NEXUS recommendations
```

**Features:**
- ✅ Reads recommendations
- ⚠️ Manual skill creation
- ⚠️ No automatic monitoring
- ✅ SOUL integration (via Python API)

---

### Cursor

**How It Works:**
- NEXUS runs in background
- Cursor reads recommendations via sidebar
- Skills created manually

**Setup:**

```bash
# 1. Setup NEXUS monitoring
*/30 * * * * /path/to/.claude/skills/scripts/nexus_auto_watch.sh

# 2. Add to workspace settings (.vscode/settings.json)
{
  "cursor.ai.systemPrompt": "Check NEXUS_RECOMMENDATIONS.md for skill recommendations. Notify user if high-priority skills are recommended."
}
```

**Usage:**

Open Cursor → AI will notify if NEXUS has recommendations → Create skills manually

**Features:**
- ⚠️ Manual recommendation checking
- ⚠️ Manual skill creation
- ❌ No automatic skill generation
- ⚠️ Limited SOUL integration

---

### Aider

**How It Works:**
- NEXUS generates recommendations
- Aider reads via file context
- Semi-automatic skill creation

**Setup:**

```bash
# 1. Setup NEXUS monitoring
*/30 * * * * /path/to/.claude/skills/scripts/nexus_auto_watch.sh

# 2. Add to .aider.conf.yml
read:
  - NEXUS_RECOMMENDATIONS.md

message: |
  Check NEXUS_RECOMMENDATIONS.md for skill recommendations.
  Create high-priority skills automatically.
```

**Usage:**

```bash
# Start Aider
aider

# Aider automatically reads NEXUS_RECOMMENDATIONS.md
# Ask to create skills:
> Create the skills recommended by NEXUS
```

**Features:**
- ✅ Automatic recommendation reading
- ⚠️ Semi-automatic skill creation
- ✅ SOUL integration
- ✅ Good file context management

---

## Cross-LLM Workflows

### Scenario 1: Claude Generates, GPT Uses

```bash
# Developer A uses Claude Code
# NEXUS auto-generates api-optimizer skill

# Developer B uses GPT
# Checks out same repository
# GPT can't load skills automatically but can read:
gpt
> Read .claude/skills/api-optimizer/SKILL.md and apply its patterns
```

### Scenario 2: Shared Recommendations

```bash
# Project team uses different LLMs
# NEXUS generates universal recommendations

# All team members can:
cat NEXUS_RECOMMENDATIONS.md

# Each LLM handles skill creation differently:
# - Claude: Automatic
# - GPT: Manual via skill-creator
# - Gemini: Manual via skill-creator
```

---

## LLM Comparison Table

| Feature | Claude Code | GPT CLI | Gemini CLI | Cursor | Aider |
|---------|-------------|---------|------------|--------|-------|
| Auto skill generation | ✅ Full | ❌ No | ❌ No | ❌ No | ⚠️ Semi |
| Read recommendations | ✅ Auto | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual | ✅ Auto |
| SOUL integration | ✅ Full | ⚠️ API only | ⚠️ API only | ❌ No | ⚠️ API only |
| Skill loading | ✅ Auto | ❌ No | ❌ No | ❌ No | ❌ No |
| Progressive disclosure | ✅ Full | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual |
| Monitoring setup | ✅ Easy | ✅ Easy | ✅ Easy | ✅ Easy | ✅ Easy |

**Legend:**
- ✅ Full support
- ⚠️ Partial / Manual
- ❌ Not supported

---

## Recommendation Consumption

### For Humans

Read the markdown file directly:

```bash
cat NEXUS_RECOMMENDATIONS.md
```

### For Claude Code

Claude automatically:
1. Runs NEXUS in background
2. Generates skills when patterns detected
3. Loads skills automatically

### For Other LLMs

**Option 1: Manual Creation**

```bash
# 1. Read recommendations
cat NEXUS_RECOMMENDATIONS.md

# 2. Ask LLM to create skill
your-llm
> Create a skill called api-optimizer based on NEXUS recommendations
```

**Option 2: Use skill-creator**

```bash
# 1. Read recommendations
python .claude/skills/nexus/scripts/nexus_analyzer.py

# 2. Use skill-creator meta-skill
your-llm
> Use skill-creator to generate the api-optimizer skill
```

**Option 3: Python Script**

```python
#!/usr/bin/env python3
"""Auto-create skills from NEXUS for non-Claude LLMs"""
import subprocess
import json

# Read recommendations
with open("NEXUS_RECOMMENDATIONS.md") as f:
    content = f.read()

# Parse recommendations (simplified)
if "🔴 CRITICAL" in content or "🟠 HIGH" in content:
    # Trigger skill-creator
    subprocess.run([
        "your-llm-cli",
        "--prompt",
        "Create skills based on NEXUS_RECOMMENDATIONS.md"
    ])
```

---

## JSON Format for Programmatic Access

Generate machine-readable format:

```bash
python .claude/skills/nexus/scripts/nexus_analyzer.py --format json
```

```json
{
  "recommendations": [
    {
      "skill_name": "api-optimizer",
      "priority": "critical",
      "pattern_type": "api_call",
      "frequency": 3.4,
      "contexts": [...]
    }
  ]
}
```

Parse with any language:

```python
# Python
import json
with open("NEXUS_RECOMMENDATIONS.json") as f:
    recs = json.load(f)

# JavaScript
const recs = require('./NEXUS_RECOMMENDATIONS.json');

# Ruby
require 'json'
recs = JSON.parse(File.read('NEXUS_RECOMMENDATIONS.json'))
```

---

## Best Practices by LLM

### For Claude Code Users

**Do:**
- ✅ Setup automatic monitoring (cron/git hooks)
- ✅ Let NEXUS auto-generate skills
- ✅ Trust the automation

**Don't:**
- ❌ Manually create skills that NEXUS recommends
- ❌ Disable auto-generation without reason

### For GPT Users

**Do:**
- ✅ Check `NEXUS_RECOMMENDATIONS.md` regularly
- ✅ Use skill-creator to generate recommended skills
- ✅ Integrate SOUL API in custom scripts

**Don't:**
- ❌ Ignore high-priority recommendations
- ❌ Recreate skills that NEXUS would auto-generate

### For Gemini Users

**Do:**
- ✅ Setup monitoring (NEXUS runs same way)
- ✅ Read recommendations at session start
- ✅ Create skills for critical/high priorities

**Don't:**
- ❌ Skip checking recommendations
- ❌ Create skills for low-priority patterns

### For Cursor Users

**Do:**
- ✅ Add NEXUS to workspace awareness
- ✅ Review recommendations weekly
- ✅ Create high-priority skills manually

**Don't:**
- ❌ Ignore NEXUS entirely (limited value)

### For Aider Users

**Do:**
- ✅ Add NEXUS_RECOMMENDATIONS.md to read list
- ✅ Configure auto-reading in .aider.conf.yml
- ✅ Batch-create recommended skills

**Don't:**
- ❌ Skip configuration step
- ❌ Ignore semi-automation opportunities

---

## Troubleshooting by LLM

### GPT: Can't Read Recommendations

**Problem:**
GPT doesn't automatically check NEXUS_RECOMMENDATIONS.md

**Solution:**
```bash
# Explicitly ask GPT:
> Read NEXUS_RECOMMENDATIONS.md and tell me what skills are recommended
```

### Gemini: Skill Creation Failed

**Problem:**
Gemini doesn't understand skill structure

**Solution:**
```bash
# Provide skill-creator guidance:
> Use the skill-creator skill to generate the api-optimizer skill following the template in .claude/skills/skill-creator/
```

### Cursor: No Skill Awareness

**Problem:**
Cursor doesn't load Claude skills

**Solution:**
```bash
# Manual skill application:
> Read .claude/skills/api-optimizer/SKILL.md and apply those patterns to this API call
```

---

## See Also

- [INSTALLATION.md](INSTALLATION.md) - Setup guide
- [MANUAL_USAGE.md](MANUAL_USAGE.md) - Command-line options
- [EXAMPLES.md](EXAMPLES.md) - Real-world examples
- SOUL [MULTI_LLM.md](../../soul/references/MULTI_LLM.md) - SOUL multi-LLM guide
- Main SKILL.md - NEXUS overview
