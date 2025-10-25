# SOUL Usage Examples

## Example 1: Basic session documentation

After completing some development work:

```bash
# Analyze current session
python scripts/trace_session.py --verbose

# Generate handoff notes
python scripts/handoff_generator.py --both
```

**Result**: Creates comprehensive logs of what was accomplished.

## Example 2: Starting a new session

When beginning work on an existing project:

```bash
# Check if previous session logs exist
ls -la .agent_*

# If they exist, review them:
cat .agent_handoff.md
cat .agent_log.md
```

**Result**: Understand previous work and next steps.

## Example 3: Custom configuration

Create `.soul_config.json`:

```json
{
  "log_level": "detailed",
  "include_git_diffs": true,
  "max_log_entries": 100,
  "custom_templates": true
}
```

Then run with custom settings:

```bash
python scripts/trace_session.py --config .soul_config.json
```

## Example 4: Multi-agent handoff

Agent A completes work:
```bash
python scripts/handoff_generator.py --detailed
```

Agent B starts new session:
```bash
# Reviews .agent_handoff.md
# Continues work
# Documents new progress
python scripts/trace_session.py --append
```

## Example 5: Problem tracking

When solving complex issues:

```bash
# Document problem-solving process
python scripts/trace_session.py --focus-problems --verbose
```

**Result**: Detailed analysis of problems encountered and solutions implemented.

## Example 6: Cross-model compatibility

For use with different AI models:

```bash
# Generate universal format logs
python scripts/trace_session.py --universal-format
python scripts/handoff_generator.py --cross-model
```

**Result**: Logs compatible with GPT, Claude, Gemini, etc.