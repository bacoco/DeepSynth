# Universal LLM Integration

SOUL works with any AI model through different integration methods.

## Claude (Skills System)

Automatic activation - no setup needed.

## GPT (Custom Instructions)

Add to your GPT custom instructions:

```
You have SOUL memory system. At session start, read:
- .agent_log.md (work history)
- .agent_handoff.md (next steps)

At session end, run:
python trace_session.py --verbose
```

## Gemini (System Prompt)

Include in your system prompt:

```
INITIALIZATION: Read .agent_log.md and .agent_handoff.md for context
TERMINATION: Run python trace_session.py before ending
```

## LLaMA/Ollama (Local Models)

```bash
# Add to your prompt
echo "You have SOUL memory. Read .agent_log.md for context. 
Run python trace_session.py when done." > soul_prompt.txt

ollama run llama2 "$(cat soul_prompt.txt) + your task"
```

## API Integration

```python
# Load SOUL context before API calls
with open('.agent_handoff.md', 'r') as f:
    soul_context = f.read()

# Include in system message
system_message = f"""
Previous session context:
{soul_context}

[Your normal prompt here]

Remember to run python trace_session.py when done.
"""
```

## Installation for other models

```bash
# Universal installation
./install.sh --model gpt       # For GPT
./install.sh --model gemini    # For Gemini  
./install.sh --model universal # For any LLM
```