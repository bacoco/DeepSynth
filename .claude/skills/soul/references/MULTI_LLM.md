# SOUL Multi-LLM Support

SOUL works with Claude Code, GPT, Gemini, and other CLI-based LLMs.

## Universal Format

SOUL uses LLM-agnostic formats:
- **Markdown** for human-readable logs
- **JSON** for machine-readable state
- **Python API** for programmatic access

No LLM-specific syntax or dependencies.

## Supported LLMs

### Claude Code (Primary)

**Installation:**
```bash
cd .claude/skills/soul/scripts
./install.sh
```

**How It Works:**
- Claude reads `.agent_handoff.md` at session start
- Git hooks automatically update SOUL files
- Skills use SOUL API for inter-skill communication

**Skill Loading:**
```bash
claude --skills soul,nexus
```

### OpenAI GPT (CLI)

**Installation:**

1. Install SOUL git hooks:
   ```bash
   cd .claude/skills/soul/scripts
   ./install.sh
   ```

2. Create GPT config in `~/.config/gpt/config.yaml`:
   ```yaml
   system_prompt: |
     You have access to SOUL session memory.
     At the start of each session, read .agent_handoff.md
     for context about previous work.

     SOUL files:
     - .agent_handoff.md - Quick start for this session
     - .agent_log.md - Detailed session history
     - .agent_status.json - Current state

     Update SOUL at the end of significant work:
     python3 .claude/skills/soul/scripts/trace_session.py
   ```

**Usage:**
```bash
# Start session
gpt

# At session start, GPT reads handoff
cat .agent_handoff.md

# After completing work
python3 .claude/skills/soul/scripts/trace_session.py
```

### Google Gemini (CLI)

**Installation:**

1. Install SOUL git hooks:
   ```bash
   cd .claude/skills/soul/scripts
   ./install.sh
   ```

2. Add to your Gemini CLI config:
   ```json
   {
     "pre_session_commands": [
       "cat .agent_handoff.md"
     ],
     "post_session_commands": [
       "python3 .claude/skills/soul/scripts/trace_session.py"
     ]
   }
   ```

**Manual Usage:**
```bash
# Start session
gemini

# Ask Gemini to read handoff
> Read .agent_handoff.md and tell me where we left off

# After work, update SOUL
> Run trace_session.py to update SOUL
```

### Cursor / Copilot

**Installation:**

1. Install git hooks:
   ```bash
   cd .claude/skills/soul/scripts
   ./install.sh
   ```

2. Git hooks will automatically track your work

**Manual Handoff:**
```bash
# Before context switch
python3 .claude/skills/soul/scripts/handoff_generator.py

# Start new session, read handoff
cat .agent_handoff.md
```

### Aider

**Installation:**

1. Install SOUL:
   ```bash
   cd .claude/skills/soul/scripts
   ./install.sh
   ```

2. Add to `.aider.conf.yml`:
   ```yaml
   auto-commits: true
   read:
     - .agent_handoff.md

   message: |
     You have access to SOUL memory system.
     Check .agent_handoff.md for session context.
   ```

**Usage:**

Aider will automatically:
- Read `.agent_handoff.md` at session start
- Trigger git hooks on commits (updating SOUL)
- Maintain session continuity

### Generic LLM CLI

For any LLM with file access:

1. **Install git hooks:**
   ```bash
   cd .claude/skills/soul/scripts
   ./install.sh
   ```

2. **At session start**, ask LLM to:
   ```
   Read .agent_handoff.md and summarize what was done previously
   ```

3. **During session**, LLM can:
   ```
   Read .agent_log.md for detailed history
   Read .agent_status.json for current state
   ```

4. **At session end**, update SOUL:
   ```bash
   python3 .claude/skills/soul/scripts/trace_session.py
   ```

---

## LLM-Specific Instructions

### For Claude Code

No special instructions needed - SOUL works natively.

### For GPT

Add this to your prompts:

```
Before starting, check .agent_handoff.md for session context.

After completing work:
- Update git
- Run: python3 .claude/skills/soul/scripts/trace_session.py
```

### For Gemini

Add this to your prompts:

```
Session memory is in .agent_handoff.md. Read it first.

When making commits, SOUL will automatically update via git hooks.

To manually update SOUL:
python3 .claude/skills/soul/scripts/trace_session.py
```

### For Cursor

Add to workspace settings (`.vscode/settings.json`):

```json
{
  "cursor.ai.systemPrompt": "Check .agent_handoff.md at session start for previous context. SOUL tracks sessions automatically via git hooks."
}
```

---

## Python API Usage (All LLMs)

All LLMs can use SOUL's Python API:

```python
import sys
from pathlib import Path

# Add SOUL to path
soul_scripts = Path(".claude/skills/soul/scripts")
sys.path.insert(0, str(soul_scripts))

from soul_api import add_soul_event, get_soul_memory

# Record events
add_soul_event("api_call", "Called external API")

# Query memory
events = get_soul_memory(filter_type="api_call")
```

No LLM-specific code required.

---

## File Compatibility

SOUL files use universal formats:

### `.agent_handoff.md`

Plain markdown - readable by any LLM:

```markdown
# Agent Handoff - 2025-10-26

## Status
- 5 files changed
- Tests passing
- On branch: feature/new-api

## Recent Work
- Implemented user authentication
- Added JWT token support

## Next Steps
1. Add refresh token logic
2. Write integration tests
```

### `.agent_log.md`

Timestamped markdown log:

```markdown
# Agent Session Log

## 2025-10-26 09:30

**Task**: Implement authentication
**Files Modified**: auth.py, tokens.py
**Outcome**: Completed - tests passing
```

### `.agent_status.json`

Standard JSON - parseable by any language:

```json
{
  "current_branch": "feature/new-api",
  "files_changed": 5,
  "last_commit": "feat: Add authentication",
  "events": [
    {
      "timestamp": "2025-10-26T09:30:15",
      "event_type": "api_call",
      "description": "Called auth service"
    }
  ]
}
```

---

## Cross-LLM Workflows

### Scenario: GPT starts, Claude finishes

**Developer using GPT:**
```bash
# Work with GPT
gpt
> Implement new feature X

# GPT commits work, git hooks update SOUL
git commit -m "WIP: Feature X partially done"

# SOUL files are automatically updated
```

**Developer using Claude:**
```bash
# Switch to Claude
claude

# Claude automatically reads .agent_handoff.md
# Sees what GPT did, continues seamlessly
```

SOUL enables cross-LLM handoffs!

---

## Team Collaboration

Different team members can use different LLMs:

```
Developer A (Claude) → commits with SOUL → pushes
Developer B (GPT) → pulls, reads SOUL → continues
Developer C (Gemini) → pulls, reads SOUL → continues
```

SOUL provides universal context format.

---

## Limitations by LLM

| Feature | Claude Code | GPT CLI | Gemini CLI | Cursor | Aider |
|---------|-------------|---------|------------|--------|-------|
| Auto handoff read | ✅ Native | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual | ✅ Config |
| Git hook auto-update | ✅ | ✅ | ✅ | ✅ | ✅ |
| Python API access | ✅ | ✅ | ✅ | ⚠️ Limited | ✅ |
| Skill-to-skill communication | ✅ Full | ⚠️ Manual | ⚠️ Manual | ❌ No | ⚠️ Limited |
| NEXUS integration | ✅ Full | ⚠️ Partial | ⚠️ Partial | ❌ No | ⚠️ Partial |

**Legend:**
- ✅ Full support
- ⚠️ Works with manual steps
- ❌ Not supported

---

## Best Practices for Multi-LLM

1. **Always commit SOUL files** - Enables cross-LLM handoffs
2. **Standardize event types** - Use consistent naming across LLMs
3. **Use Python API when possible** - More reliable than manual updates
4. **Read handoff at session start** - Even if LLM doesn't do it automatically
5. **Update SOUL at session end** - `trace_session.py` before context switch

---

## Troubleshooting

### LLM doesn't see SOUL files

**Solution:** Explicitly ask LLM to read them:
```
Read .agent_handoff.md and tell me what was done previously
```

### Git hooks not triggering

**Solution:** Reinstall hooks:
```bash
cd .claude/skills/soul/scripts
./install.sh
```

Check `.git/hooks/post-commit` exists and is executable.

### Python API import errors

**Solution:** Add SOUL to Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".claude/skills/soul/scripts")))
```

### SOUL files conflict on merge

**Solution:** SOUL files are auto-generated, keep latest:
```bash
git checkout --theirs .agent_log.md
git checkout --theirs .agent_status.json
git checkout --theirs .agent_handoff.md
```

Or regenerate after merge:
```bash
python3 .claude/skills/soul/scripts/trace_session.py
```

---

## Extending SOUL for Custom LLMs

To add support for a new LLM:

1. **Install git hooks** - Universal across all LLMs
2. **Configure LLM** - Add SOUL file reading to system prompt
3. **Add to documentation** - Submit PR to add your LLM config

Example contribution:

```markdown
### MyNewLLM

**Installation:**
1. Install git hooks: `cd .claude/skills/soul/scripts && ./install.sh`
2. Add to `~/.mynewllm/config`:
   ```
   pre_session: cat .agent_handoff.md
   ```

**Usage:**
```bash
mynewllm
# Automatically reads SOUL handoff
```
```

---

## See Also

- [API_REFERENCE.md](API_REFERENCE.md) - Complete API documentation
- [WORKFLOWS.md](WORKFLOWS.md) - Common usage patterns
- Main SKILL.md - SOUL overview
