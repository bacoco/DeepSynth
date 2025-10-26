---
name: nexus
description: Unified analyzer that monitors SOUL memory, PRD files, tasks, and code to automatically recommend skills for generation. The brain of the skill ecosystem.
---

# NEXUS - Universal Skill Recommendation Engine

**Analyzes everything and tells Claude which skills to generate.**

NEXUS is the brain of the skill ecosystem. It watches SOUL, reads your PRD, analyzes your tasks, and recommends exactly which skills you need.

---

## What NEXUS does

NEXUS performs unified analysis from multiple sources:

### 1. **SOUL Memory Analysis**
- Reads `.agent_log.md` and `.agent_status.json`
- Detects recurring patterns (API calls, data processing, etc.)
- Identifies patterns that appear 5+ times
- Calculates priority based on frequency

### 2. **PRD Analysis**
- Scans for PRD files (`*PRD*.md`, `*REQUIREMENTS*.md`, `*ROADMAP*.md`)
- Extracts tasks and requirements
- Classifies by domain (api, testing, deployment, etc.)
- Counts tasks per domain

### 3. **Task Analysis**
- Reads TODO files and task lists
- Parses checkboxes, numbered lists, bullets
- Groups related tasks
- Identifies skill needs

### 4. **Code Analysis** (future)
- Analyzes existing codebase structure
- Detects missing test coverage
- Identifies documentation gaps

---

## Output: NEXUS_RECOMMENDATIONS.md

NEXUS generates a markdown file with prioritized skill recommendations:

```markdown
# NEXUS Skill Recommendations

## Summary
- Total recommendations: 5
- High priority: 2
- Medium priority: 2
- Low priority: 1

## Recommended Skills

### 1. 🔴 api-optimizer (CRITICAL)
**Pattern:** api_call
**Frequency:** 3.5 times/day (24 total)
**Reason:** Detected 24 API operations in 7 days
**Capabilities:**
- Rate limiting and retry logic
- Error handling patterns
- Response caching

### 2. 🟠 test-guardian (HIGH)
**Pattern:** testing
**Tasks:** 12 test-related tasks in PRD
**Reason:** Large testing requirements
...
```

---

## When Claude activates NEXUS

NEXUS runs:
- **Automatically**: Every 24 hours (configurable)
- **On demand**: When you ask "What skills should I generate?"
- **After PRD changes**: When you update requirements
- **When SOUL detects high-frequency patterns**

---

## Workflow

```
User works on project
        ↓
SOUL traces everything
        ↓
NEXUS analyzes periodically:
  - SOUL memory (patterns)
  - PRD files (requirements)
  - Task lists (TODO)
  - Code (optional)
        ↓
Generates NEXUS_RECOMMENDATIONS.md
        ↓
Claude reads recommendations
        ↓
You: "Generate the api-optimizer skill"
        ↓
skill-generator creates it
        ↓
New skill ready!
```

---

## Configuration

Create `.nexus_config.json`:

```json
{
  "analysis": {
    "threshold": 5,
    "window_days": 7,
    "auto_run": false,
    "auto_run_interval_hours": 24
  },
  "sources": {
    "soul_memory": true,
    "prd_files": true,
    "task_lists": true,
    "code_analysis": false
  },
  "output": {
    "file": "NEXUS_RECOMMENDATIONS.md",
    "format": "markdown",
    "include_examples": true
  }
}
```

---

## Manual execution

```bash
# Run full analysis
python .claude/skills/nexus/scripts/nexus_analyzer.py

# Custom parameters
python .claude/skills/nexus/scripts/nexus_analyzer.py \
  --threshold 3 \
  --days 14 \
  --output MY_RECOMMENDATIONS.md

# Specific repository
python .claude/skills/nexus/scripts/nexus_analyzer.py \
  --repo /path/to/project
```

---

## Integration with other skills

**SOUL → NEXUS:**
- NEXUS reads SOUL memory files
- Detects patterns from traced events
- Uses `soul_api.py` for pattern analysis

**NEXUS → skill-generator:**
- NEXUS writes recommendations
- skill-generator reads recommendations
- Creates skills automatically

**Complete flow:**
```
SOUL (traces) → NEXUS (analyzes) → skill-generator (creates) → New Skills → SOUL (uses)
```

---

## Priority levels

NEXUS assigns priorities based on:

**CRITICAL** 🔴
- Pattern appears 3+ times/day
- OR 20+ task items in PRD
- Immediate action recommended

**HIGH** 🟠
- Pattern appears 1-3 times/day
- OR 10-20 task items in PRD
- Should generate soon

**MEDIUM** 🟡
- Pattern appears 3-7 times/week
- OR 5-10 task items in PRD
- Generate when convenient

**LOW** 🟢
- Pattern appears <3 times/week
- OR <5 task items in PRD
- Optional, monitor for increase

---

## Example recommendations

### From SOUL patterns:
```
Detected 24 API calls in 7 days
→ Recommend "api-optimizer" skill (HIGH)
  Capabilities: rate limiting, retry logic, caching
```

### From PRD analysis:
```
Found 15 testing tasks in PRD
→ Recommend "test-guardian" skill (HIGH)
  Capabilities: test generation, coverage analysis
```

### From task lists:
```
Found 8 deployment-related TODOs
→ Recommend "deploy-sage" skill (MEDIUM)
  Capabilities: CI/CD, Docker, environment management
```

### Combined (SOUL + PRD):
```
Detected 10 data transformations (SOUL)
+ 7 ETL tasks (PRD)
= Recommend "data-wizard" skill (CRITICAL)
  Capabilities: CSV/JSON parsing, validation, transformations
```

---

## Multi-LLM support

NEXUS works with:
- ✅ **Claude Code**: Native integration
- ✅ **GPT/Codex**: Reads NEXUS_RECOMMENDATIONS.md
- ✅ **Gemini CLI**: Reads NEXUS_RECOMMENDATIONS.md

The output file is universal - any LLM can read and understand it.

---

## Advanced features

### Pattern merging
If SOUL detects "api_call" pattern AND PRD has "API tasks", NEXUS merges them:
```
SOUL: 12 API calls
PRD:  8 API tasks
= Combined priority CRITICAL (instead of separate HIGH)
```

### Duplicate detection
NEXUS won't recommend skills that already exist:
```
Detected "testing" pattern
→ Check: does "test-guardian" exist?
→ Yes? Skip recommendation
→ No? Add to recommendations
```

### Context preservation
Recommendations include example contexts:
```
Example API calls from SOUL:
- GET /users (called 8 times)
- POST /auth (called 6 times)
- GET /data (called 10 times)
```

---

## Why NEXUS is generic

Unlike specific skills (api-master, db-handler, etc.), NEXUS:
- ✅ Doesn't solve ONE problem
- ✅ Solves the meta-problem: "what skills do you need?"
- ✅ Adapts to ANY project type
- ✅ Learns from YOUR specific patterns
- ✅ Never becomes obsolete

---

## Part of the ecosystem

**SOUL** = Remembers everything
**NEXUS** = Analyzes and recommends
**skill-generator** = Creates skills
**Generated skills** = Solve specific problems

NEXUS is the connector that makes the whole system intelligent and self-improving.

---

*Generated: 2025-10-26*
*Part of the SOUL-NEXUS-skill-generator ecosystem*
*Universal skill recommendation engine*
