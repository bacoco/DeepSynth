---
name: soul
description: Universal memory system that automatically tracks all sessions, enables agent handoffs, and provides inter-skill communication through persistent logs and machine-readable state files.
---

# SOUL - Session Orchestration & Universal Logging

SOUL is an automatic memory system that traces all work across sessions and enables skills to communicate with each other.

## What SOUL Provides

1. **Automatic session tracing** - Tracks all file changes, git operations, and major events
2. **Agent handoffs** - Seamless context transfer between sessions
3. **Inter-skill communication** - Python API for skills to share data and patterns
4. **Pattern detection** - Analyzes work patterns to recommend new skills (via NEXUS)

## Core Files Generated

SOUL automatically maintains three files in your project root:

- **`.agent_log.md`** - Human-readable session history with detailed narrative
- **`.agent_status.json`** - Machine-readable current state and metrics
- **`.agent_handoff.md`** - Quick start guide for next session

These files are automatically updated throughout your work session.

## How It Works

### Automatic Tracing

SOUL runs automatically via git hooks. Every time you:
- Make file changes
- Commit code
- Complete a task

SOUL updates the memory files with context about what happened and why.

### Agent Handoffs

When starting a new session, Claude reads `.agent_handoff.md` to understand:
- What was done in the last session
- Current repository state
- Priority next steps
- Recent work context

### Inter-Skill Communication

Skills can use SOUL's Python API to communicate:

```python
from soul_api import add_soul_event, get_soul_memory, get_pattern_analysis

# Record an event
add_soul_event("api_call", "Called GitHub API", {
    "endpoint": "/repos/user/repo",
    "status": 200
})

# Query memory
recent_events = get_soul_memory(filter_type="api_call", limit=10)

# Analyze patterns (used by NEXUS)
patterns = get_pattern_analysis(days=7, threshold=5)
```

See [API_REFERENCE.md](references/API_REFERENCE.md) for complete documentation.

## Installation

Run the installation script to set up git hooks:

```bash
cd .claude/skills/soul/scripts
./install.sh
```

This configures SOUL to automatically track your work.

## Usage

### Manual Session Tracing

To manually update SOUL files:

```bash
python3 .claude/skills/soul/scripts/trace_session.py
```

### Manual Handoff Generation

To generate a handoff document:

```bash
python3 .claude/skills/soul/scripts/handoff_generator.py
```

### Using SOUL API in Your Skills

1. Import the API:
   ```python
   from soul_api import add_soul_event, get_soul_memory
   ```

2. Record events during your skill's operation
3. Query patterns to inform decisions
4. SOUL handles all file locking and persistence

## Integration with NEXUS

SOUL's pattern analysis powers NEXUS automatic skill generation:

1. SOUL tracks recurring patterns (API calls, data processing, etc.)
2. NEXUS queries SOUL for patterns above a threshold
3. NEXUS automatically generates skills when patterns reach critical frequency
4. New skills use SOUL API to record their own events

This creates a self-improving system where skills emerge from actual usage patterns.

## Files and Scripts

### Scripts

- **`trace_session.py`** - Updates `.agent_log.md` and `.agent_status.json`
- **`handoff_generator.py`** - Creates `.agent_handoff.md`
- **`soul_api.py`** - Python API for inter-skill communication
- **`install.sh`** - Sets up git hooks

### References

- **[API_REFERENCE.md](references/API_REFERENCE.md)** - Complete SOUL API documentation
- **[WORKFLOWS.md](references/WORKFLOWS.md)** - Common SOUL usage patterns
- **[MULTI_LLM.md](references/MULTI_LLM.md)** - Using SOUL with GPT, Gemini, etc.

## Best Practices

1. **Let it run automatically** - Don't manually trace unless needed
2. **Check `.agent_handoff.md` at session start** - Quick context refresh
3. **Use SOUL API in custom skills** - Enables automatic pattern detection
4. **Commit SOUL files with your work** - Preserves memory across machines

## Multi-LLM Support

SOUL works with Claude Code, GPT, Gemini, and other CLI-based LLMs. The memory files use universal markdown and JSON formats.

See [MULTI_LLM.md](references/MULTI_LLM.md) for LLM-specific integration guides.

---

*SOUL is the foundation of DeepSynth's self-improving skills system.*
