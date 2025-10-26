# SOUL API Reference

Complete documentation for SOUL's Python API.

## Installation

Add SOUL scripts to your Python path:

```python
import sys
from pathlib import Path

soul_scripts = Path(".claude/skills/soul/scripts")
sys.path.insert(0, str(soul_scripts))

from soul_api import *
```

## Core Functions

### `add_soul_event(event_type: str, description: str, metadata: Optional[Dict] = None)`

Record an event in SOUL memory.

**Parameters:**
- `event_type` (str): Category of event (e.g., "api_call", "data_processing", "error")
- `description` (str): Human-readable description
- `metadata` (dict, optional): Additional structured data

**Returns:** None

**Example:**

```python
add_soul_event(
    "api_call",
    "Called GitHub API to fetch repositories",
    {
        "endpoint": "/user/repos",
        "status_code": 200,
        "response_time_ms": 145
    }
)
```

**Thread Safety:** Uses file locking to prevent concurrent write conflicts.

---

### `get_soul_memory(filter_type=None, since=None, limit=None) -> List[Dict]`

Query SOUL memory with filters.

**Parameters:**
- `filter_type` (str, optional): Filter by event_type
- `since` (datetime, optional): Only events after this timestamp
- `limit` (int, optional): Maximum number of events to return

**Returns:** List of event dictionaries, sorted by timestamp (newest first)

**Event Structure:**
```python
{
    "timestamp": "2025-10-26T09:30:15",
    "event_type": "api_call",
    "description": "Called GitHub API...",
    "metadata": { ... }
}
```

**Examples:**

```python
# Get all recent events
all_events = get_soul_memory(limit=100)

# Get only API calls
api_events = get_soul_memory(filter_type="api_call")

# Get events from last 24 hours
from datetime import datetime, timedelta
yesterday = datetime.now() - timedelta(days=1)
recent = get_soul_memory(since=yesterday)

# Combine filters
recent_api = get_soul_memory(
    filter_type="api_call",
    since=yesterday,
    limit=50
)
```

---

### `get_pattern_analysis(days: int = 7, threshold: int = 5) -> Dict`

Analyze patterns in SOUL memory for skill recommendations.

**Parameters:**
- `days` (int): Number of days to analyze (default: 7)
- `threshold` (int): Minimum occurrences to consider a pattern (default: 5)

**Returns:** Dictionary with pattern analysis:

```python
{
    "patterns_detected": 2,
    "analysis_period_days": 7,
    "patterns": {
        "api_call": {
            "count": 15,
            "frequency": 2.14,  # per day
            "priority": "high",
            "suggested_skill": "api-optimizer",
            "contexts": [
                "Called GitHub API...",
                "Called Stripe API...",
                ...
            ]
        },
        "data_processing": {
            "count": 8,
            "frequency": 1.14,
            "priority": "medium",
            "suggested_skill": "data-transformer",
            "contexts": [...]
        }
    }
}
```

**Priority Calculation:**
- `critical`: frequency >= 2/day
- `high`: frequency >= 1/day OR count >= threshold * 1.5
- `medium`: count >= threshold
- `low`: count < threshold

**Example:**

```python
# Analyze last 7 days with default threshold
patterns = get_pattern_analysis()

for pattern_type, info in patterns["patterns"].items():
    print(f"{pattern_type}: {info['count']} occurrences ({info['priority']} priority)")
    print(f"  Suggested skill: {info['suggested_skill']}")

# Custom analysis: last 30 days, threshold of 10
patterns = get_pattern_analysis(days=30, threshold=10)
```

---

### `get_current_context() -> Dict`

Get current session context from `.agent_status.json`.

**Parameters:** None

**Returns:** Dictionary with current state:

```python
{
    "current_branch": "main",
    "files_changed": 5,
    "last_commit": "feat: Add SOUL API",
    "last_update": "2025-10-26T09:30:15",
    "total_sessions": 12,
    "events": [...]  # Recent events
}
```

**Example:**

```python
context = get_current_context()
print(f"Current branch: {context['current_branch']}")
print(f"Files changed: {context['files_changed']}")
```

---

### `get_soul_instance() -> SOULMemory`

Get direct access to SOUL memory instance for advanced usage.

**Parameters:** None

**Returns:** SOULMemory instance with direct file access

**Example:**

```python
soul = get_soul_instance()

# Direct file path access
print(f"Log file: {soul.log_file}")
print(f"Status file: {soul.status_file}")

# Force save current state
soul._save_status()
```

**Note:** Most use cases should use the convenience functions above rather than direct instance access.

---

## Event Type Conventions

Use consistent event types across skills for better pattern detection:

### Standard Event Types

- **`api_call`** - External API requests
- **`data_processing`** - Data transformation, parsing, ETL
- **`file_operation`** - File read/write/move operations
- **`error`** - Errors and exceptions
- **`skill_generated`** - New skill created by NEXUS
- **`session_traced`** - Session tracing completed
- **`git_operation`** - Git commands executed
- **`test_run`** - Test execution
- **`build`** - Build/compile operations
- **`deployment`** - Deploy/release operations

### Custom Event Types

You can create custom event types for domain-specific patterns:

```python
add_soul_event("pdf_processing", "Extracted text from invoice PDF", {
    "pages": 3,
    "file_size_kb": 245
})
```

If the pattern occurs frequently, NEXUS will recommend creating a dedicated skill.

---

## Thread Safety and Concurrency

SOUL uses file-based locking to prevent concurrent write conflicts:

```python
import fcntl

with open(lock_file, 'w') as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    # Safe to write to SOUL files
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
```

Multiple processes can safely call SOUL API functions simultaneously.

---

## File Locations

SOUL stores data in your project root:

- **`.agent_log.md`** - Human-readable session history
- **`.agent_status.json`** - Machine-readable current state
- **`.agent_handoff.md`** - Next session quick start
- **`.soul_lock`** - Temporary lock file (auto-cleaned)

All files are gitignored by default (see `.gitignore`).

---

## Error Handling

SOUL API functions handle errors gracefully:

```python
try:
    add_soul_event("test", "Test event")
except Exception as e:
    print(f"SOUL error: {e}")
    # Your code continues - SOUL failures don't break workflows
```

If SOUL files are corrupted or missing, functions will:
1. Log a warning
2. Attempt to recreate files
3. Return empty/default values
4. Never raise unhandled exceptions

---

## Performance Considerations

- **File I/O**: Each API call reads/writes JSON files - batch events when possible
- **Lock contention**: Under heavy concurrent access, operations may block briefly
- **Memory size**: `.agent_status.json` keeps last 1000 events - older events auto-pruned

**Best Practice:** Record significant events (API calls, errors, completions), not every function call.

---

## Integration Examples

### Using SOUL in a Custom Skill

```python
#!/usr/bin/env python3
"""
My custom skill that uses SOUL
"""
import sys
from pathlib import Path

# Add SOUL to path
soul_path = Path(".claude/skills/soul/scripts")
sys.path.insert(0, str(soul_path))

from soul_api import add_soul_event, get_soul_memory

def my_skill_function():
    # Record start
    add_soul_event("my_skill", "Starting custom processing")

    try:
        # Do work
        result = process_data()

        # Record success
        add_soul_event("my_skill", "Processing completed", {
            "items_processed": len(result)
        })

    except Exception as e:
        # Record error
        add_soul_event("error", f"My skill failed: {str(e)}", {
            "skill": "my_skill",
            "error_type": type(e).__name__
        })
        raise

# Check if this pattern has occurred before
previous_runs = get_soul_memory(filter_type="my_skill")
if len(previous_runs) > 10:
    print("⚠️ This skill is being used frequently!")
    print("   NEXUS may recommend optimizations.")
```

### Querying Patterns for Recommendations

```python
from soul_api import get_pattern_analysis

# Analyze work patterns
patterns = get_pattern_analysis(days=14, threshold=3)

print(f"Found {patterns['patterns_detected']} patterns in last 14 days:\n")

for pattern_type, info in patterns["patterns"].items():
    if info["priority"] in ["high", "critical"]:
        print(f"⚠️ {pattern_type}:")
        print(f"   Count: {info['count']}")
        print(f"   Frequency: {info['frequency']:.1f}/day")
        print(f"   Recommended skill: {info['suggested_skill']}")
        print()
```

---

## See Also

- [WORKFLOWS.md](WORKFLOWS.md) - Common SOUL usage patterns
- [MULTI_LLM.md](MULTI_LLM.md) - Using SOUL with different LLMs
- Main SKILL.md - SOUL overview and quick start
