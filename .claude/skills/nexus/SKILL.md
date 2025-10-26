---
name: nexus
description: Unified analyzer that monitors SOUL memory, PRD files, and tasks to automatically generate skills based on detected patterns. The brain of the self-improving skills system.
---

# NEXUS - Automatic Skill Generator

**Analyzes your work patterns and automatically generates the skills you need.**

NEXUS is the brain of the skill ecosystem. It watches SOUL memory, reads your PRD files, analyzes your tasks, and automatically creates new skills when patterns emerge.

## What NEXUS Does

NEXUS performs unified analysis from multiple sources:

### 1. SOUL Memory Analysis
- Reads `.agent_log.md` and `.agent_status.json`
- Detects recurring patterns (API calls, data processing, errors, etc.)
- Identifies patterns that appear ≥ threshold (default: 5 times)
- Calculates priority based on frequency

### 2. PRD Analysis
- Scans for PRD files (`*PRD*.md`, `*REQUIREMENTS*.md`, `*ROADMAP*.md`)
- Extracts tasks and requirements
- Classifies by domain (api, testing, deployment, etc.)
- Counts tasks per domain to identify skill needs

### 3. Task Analysis
- Reads TODO files and task lists
- Parses checkboxes, numbered lists, bullets
- Groups related tasks
- Identifies skill opportunities

## Automatic Skill Generation

**NEXUS runs automatically and generates skills without user intervention:**

- **Periodically**: Every 30 minutes via cron (optional)
- **On git commit**: Via post-commit hook (optional)
- **When critical patterns detected**: Immediate generation

**Skills are auto-generated when:**
- Pattern appears ≥ threshold (default: 5 times)
- Priority is high or critical
- Skill doesn't already exist

## How It Works

```
You work normally
        ↓
SOUL traces everything
        ↓
NEXUS monitors automatically:
  - SOUL memory (patterns)
  - PRD files (requirements)
  - Task lists (TODO)
        ↓
Detects patterns >= threshold
        ↓
Auto-generates skills if priority >= high:
  - Creates .claude/skills/[skill-name]/
  - Generates SKILL.md with progressive disclosure
  - Creates scripts with SOUL API integration
  - Records in SOUL memory
        ↓
New skill ready immediately!
        ↓
Claude uses it automatically
```

**No user intervention needed.**

## Priority Levels

NEXUS assigns priorities based on frequency and task count:

- **🔴 CRITICAL**: Pattern appears 2+ times/day → Auto-generate immediately
- **🟠 HIGH**: Pattern appears 1+ times/day → Auto-generate
- **🟡 MEDIUM**: Pattern appears 3-7 times/week → Monitor
- **🟢 LOW**: Pattern appears <3 times/week → Monitor

Only HIGH and CRITICAL priorities trigger automatic generation.

## Output

NEXUS generates `NEXUS_RECOMMENDATIONS.md` with prioritized skill recommendations:

```markdown
# NEXUS Skill Recommendations

## Summary
- Total recommendations: 3
- High priority: 2
- Medium priority: 1

## Recommended Skills

### 1. 🔴 api-optimizer (CRITICAL)
**Pattern:** api_call
**Frequency:** 3.5 times/day (24 total in 7 days)
**Reason:** Frequent API operations detected
...
```

See [OUTPUT_FORMAT.md](references/OUTPUT_FORMAT.md) for complete output specification.

## Quick Start

### Automatic Mode (Recommended)

Run auto-generator to analyze and generate skills automatically:

```bash
python .claude/skills/nexus/scripts/auto_skill_generator.py
```

Skills with HIGH or CRITICAL priority will be generated automatically.

### Analysis Only

Generate recommendations without auto-creating skills:

```bash
python .claude/skills/nexus/scripts/nexus_analyzer.py
```

This creates `NEXUS_RECOMMENDATIONS.md` for manual review.

### Setup Monitoring

Add to crontab for automatic periodic checks:

```bash
# Every 30 minutes
*/30 * * * * /path/to/.claude/skills/scripts/nexus_auto_watch.sh
```

Or use as git hook (see [INSTALLATION.md](references/INSTALLATION.md)).

## Integration with SOUL

NEXUS and SOUL work together seamlessly:

1. **SOUL traces** your work and records events
2. **NEXUS analyzes** SOUL memory for patterns
3. **NEXUS generates** skills when patterns reach threshold
4. **New skills use** SOUL API to record their own events
5. **Pattern detection improves** as more skills contribute data

This creates a self-improving system where skills emerge from actual usage patterns.

## Generated Skill Structure

NEXUS creates skills following best practices:

```
generated-skill/
├── SKILL.md (with YAML frontmatter)
├── scripts/
│   └── main.py (with SOUL API integration)
└── references/ (if needed)
```

All generated skills:
- ✅ Follow progressive disclosure principles
- ✅ Include SOUL API integration
- ✅ Have concise SKILL.md (<200 lines)
- ✅ Record their own events for future pattern detection
- ✅ Work with Claude Code, GPT, and Gemini

## Multi-LLM Support

NEXUS works with any CLI-based LLM:
- ✅ **Claude Code**: Native integration
- ✅ **GPT/Codex**: Reads `NEXUS_RECOMMENDATIONS.md`
- ✅ **Gemini CLI**: Reads `NEXUS_RECOMMENDATIONS.md`

See [MULTI_LLM.md](references/MULTI_LLM.md) for LLM-specific guides.

## Advanced Features

- **Pattern merging**: Combines SOUL + PRD patterns for higher priority
- **Duplicate detection**: Never generates skills that already exist
- **Context preservation**: Recommendations include example usage
- **Custom thresholds**: Configure sensitivity via command-line args

See [ADVANCED.md](references/ADVANCED.md) for detailed documentation.

## Configuration

Create `.nexus_config.json` for custom settings:

```json
{
  "analysis": {
    "threshold": 5,
    "window_days": 7
  },
  "sources": {
    "soul_memory": true,
    "prd_files": true,
    "task_lists": true
  }
}
```

See [CONFIGURATION.md](references/CONFIGURATION.md) for all options.

## References

- **[INSTALLATION.md](references/INSTALLATION.md)** - Setup and git hooks
- **[MANUAL_USAGE.md](references/MANUAL_USAGE.md)** - Command-line options
- **[OUTPUT_FORMAT.md](references/OUTPUT_FORMAT.md)** - Recommendation file format
- **[EXAMPLES.md](references/EXAMPLES.md)** - Real-world skill generation examples
- **[ADVANCED.md](references/ADVANCED.md)** - Pattern merging, customization
- **[CONFIGURATION.md](references/CONFIGURATION.md)** - Complete config reference
- **[MULTI_LLM.md](references/MULTI_LLM.md)** - Using with GPT, Gemini, etc.

## Part of the Ecosystem

**SOUL** → Remembers everything
**NEXUS** → Analyzes and generates
**Generated skills** → Solve specific problems
**Skills use SOUL** → Pattern detection improves

NEXUS makes the system intelligent and self-improving.

---

*NEXUS - The universal skill recommendation and generation engine*
