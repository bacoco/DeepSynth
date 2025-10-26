# NEXUS Advanced Features

Advanced functionality and customization options.

## Pattern Merging

NEXUS intelligently merges patterns from multiple sources to calculate combined priority.

### How It Works

When the same pattern type appears in multiple sources:

```
SOUL pattern: api_call
  Count: 12
  Frequency: 1.7/day
  Priority: HIGH

PRD pattern: api_call
  Task count: 8
  Priority: MEDIUM

MERGED:
  Source: SOUL + PRD
  Combined priority: CRITICAL
  Reason: Pattern in both usage and requirements
```

### Merging Rules

**Priority Escalation:**

| SOUL Priority | PRD Priority | Merged Priority |
|---------------|--------------|-----------------|
| HIGH | HIGH | CRITICAL |
| HIGH | MEDIUM | CRITICAL |
| HIGH | LOW | HIGH |
| MEDIUM | MEDIUM | HIGH |
| MEDIUM | LOW | MEDIUM |
| LOW | LOW | MEDIUM |

**Additional Factors:**

- **Trend analysis**: Increasing frequency → priority boost
- **Error correlation**: Errors in same domain → priority boost
- **Task complexity**: PRD task complexity → priority adjustment

### Example

```python
# Pattern appears in SOUL
soul_pattern = {
    "type": "data_processing",
    "count": 10,
    "frequency": 1.4,  # per day
    "priority": "medium"
}

# Same pattern in PRD
prd_pattern = {
    "type": "data_processing",
    "task_count": 6,
    "priority": "medium"
}

# NEXUS merges:
merged = {
    "type": "data_processing",
    "source": "SOUL + PRD",
    "priority": "high",  # Escalated from medium
    "soul_frequency": 1.4,
    "prd_tasks": 6,
    "reason": "Pattern confirmed in both usage and requirements"
}
```

---

## Duplicate Detection

NEXUS prevents generating duplicate skills.

### Detection Methods

1. **Exact name match**: Skill directory already exists
2. **Pattern match**: Existing skill covers the pattern type
3. **Capability match**: Existing skill provides same capabilities

### Example

```bash
# Existing skills
.claude/skills/
├── api-optimizer/     # Handles api_call patterns
├── test-guardian/     # Handles testing patterns
└── data-master/       # Handles data_processing patterns

# NEXUS detects api_call pattern
# Checks: does api-optimizer exist? YES
# Result: Skip recommendation
```

### Override

Force recommendation even if skill exists:

```bash
python auto_skill_generator.py --force-recommendations
```

---

## Context Preservation

NEXUS preserves example contexts from SOUL for better skill generation.

### How Contexts Are Collected

```python
# SOUL events with metadata
events = [
    {
        "type": "api_call",
        "description": "Called GitHub API for repos",
        "metadata": {"endpoint": "/user/repos", "status": 200}
    },
    {
        "type": "api_call",
        "description": "Called Stripe API for customers",
        "metadata": {"endpoint": "/customers", "status": 200}
    },
    # ... more events
]

# NEXUS extracts contexts
contexts = [
    "Called GitHub API for repos",
    "Called Stripe API for customers",
    # ... up to include_contexts limit (default: 5)
]
```

### Usage in Skill Generation

Generated skills include these contexts as examples:

```markdown
# api-optimizer

## Common Use Cases

Based on analyzed patterns:

- Calling GitHub API for repository data
- Calling Stripe API for customer information
- Calling SendGrid API for email delivery

## Example Code

\`\`\`python
# Pattern: GitHub API calls
response = call_api("github.com", "/user/repos", retry=True)
\`\`\`
```

---

## Trend Analysis

NEXUS tracks pattern trends over time.

### How It Works

```python
# Analyze pattern frequency over time windows
week_1_frequency = 1.2  # calls/day
week_2_frequency = 1.8  # calls/day
week_3_frequency = 2.4  # calls/day
week_4_frequency = 3.1  # calls/day

# Calculate trend
trend = "increasing"  # 30%+ increase
priority_boost = True  # Apply boost for increasing trends
```

### Priority Adjustments

| Trend | Priority Adjustment |
|-------|-------------------|
| Rapidly increasing (>50%/week) | +2 levels (medium → critical) |
| Increasing (>30%/week) | +1 level (medium → high) |
| Stable (±30%) | No change |
| Decreasing (>30%/week) | -1 level (high → medium) |

### Example

```markdown
### 🔴 database-optimizer (CRITICAL)

**Trend:** Rapidly increasing ⬆️
- Week 1: 1.5 queries/day
- Week 2: 2.3 queries/day
- Week 3: 3.8 queries/day
- Week 4: 6.2 queries/day

**Reason:** Pattern frequency increasing 40% per week
**Priority:** Escalated to CRITICAL due to rapid growth
```

---

## Custom Pattern Definitions

Define custom patterns in configuration.

### Example Configuration

```json
{
  "patterns": {
    "custom_patterns": {
      "pdf_processing": {
        "keywords": ["pdf", "extract", "parse pdf", "pdfplumber"],
        "suggested_skill": "pdf-master",
        "priority_multiplier": 1.5,
        "required_metadata": ["file_size", "pages"]
      },
      "image_manipulation": {
        "keywords": ["image", "resize", "crop", "convert", "pillow"],
        "suggested_skill": "image-wizard",
        "priority_multiplier": 2.0,
        "capabilities": [
          "Image resizing and cropping",
          "Format conversion",
          "Batch processing",
          "Optimization"
        ]
      }
    }
  }
}
```

### How Custom Patterns Work

1. NEXUS scans SOUL events for keywords
2. Matches events to custom pattern definitions
3. Applies priority multiplier
4. Uses suggested skill name
5. Includes predefined capabilities in recommendations

---

## Skill Template Customization

Customize generated skill structure.

### Template Variables

```markdown
# {skill_name}

**Pattern Type:** {pattern_type}
**Generated:** {timestamp}

## Detected Use Cases

{context_examples}

## Capabilities

{suggested_capabilities}

## Integration with SOUL

{soul_api_code}
```

### Create Custom Template

```bash
# Create template directory
mkdir -p .nexus_templates/

# Create custom template
cat > .nexus_templates/my_template.md << 'EOF'
---
name: {skill_name}
description: {description}
---

# {skill_name}

Auto-generated skill for {pattern_type} patterns.

## Quick Start

{quick_start_code}

## Examples

{context_examples}
EOF

# Use custom template
python auto_skill_generator.py --template my_template
```

---

## Integration with External Tools

### GitHub Actions

```yaml
# .github/workflows/nexus.yml
name: NEXUS Monitoring

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 */6 * * *'

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Run NEXUS Analysis
        run: |
          python .claude/skills/nexus/scripts/auto_skill_generator.py \
            --dry-run \
            --output nexus-report.md

      - name: Create Issue if High Priority Found
        run: |
          if grep -q "🔴 CRITICAL" nexus-report.md; then
            gh issue create \
              --title "NEXUS: Critical skill recommendation" \
              --body-file nexus-report.md \
              --label "nexus,auto-generated"
          fi
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### Slack Notifications

```python
#!/usr/bin/env python3
"""Send NEXUS recommendations to Slack"""
import requests
import json

# Run analysis
from auto_skill_generator import AutoSkillGenerator

generator = AutoSkillGenerator()
results = generator.run_auto_generation()

# Send to Slack if high-priority recommendations
if results["high_priority_count"] > 0:
    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

    message = {
        "text": f"🤖 NEXUS found {results['high_priority_count']} high-priority skill recommendations",
        "attachments": [
            {
                "color": "danger",
                "fields": [
                    {
                        "title": "Recommendations",
                        "value": "\n".join(results["skill_names"]),
                        "short": False
                    }
                ]
            }
        ]
    }

    requests.post(webhook_url, json=message)
```

### Jira Integration

```python
#!/usr/bin/env python3
"""Create Jira tickets for NEXUS recommendations"""
from jira import JIRA
from auto_skill_generator import AutoSkillGenerator

# Connect to Jira
jira = JIRA(server="https://your-domain.atlassian.net", auth=("email", "token"))

# Run NEXUS
generator = AutoSkillGenerator()
recommendations = generator.get_recommendations()

# Create tickets for high-priority recommendations
for rec in recommendations:
    if rec["priority"] in ["critical", "high"]:
        issue = jira.create_issue(
            project="SKILLS",
            summary=f"Implement {rec['skill_name']} skill",
            description=rec["reason"],
            issuetype={"name": "Task"},
            labels=["nexus", "auto-generated", rec["priority"]]
        )
        print(f"Created: {issue.key}")
```

---

## Performance Optimization

### Caching Analysis Results

```python
#!/usr/bin/env python3
"""Cache NEXUS analysis for faster runs"""
import pickle
from pathlib import Path
from datetime import datetime, timedelta

CACHE_FILE = ".nexus_cache.pkl"
CACHE_TTL_HOURS = 1

def get_cached_analysis():
    if not Path(CACHE_FILE).exists():
        return None

    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)

    if datetime.now() - cache["timestamp"] < timedelta(hours=CACHE_TTL_HOURS):
        return cache["data"]

    return None

def save_analysis_cache(data):
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({"timestamp": datetime.now(), "data": data}, f)
```

### Incremental Analysis

```python
# Only analyze new events since last run
last_run_time = load_last_run_timestamp()
new_events = get_soul_memory(since=last_run_time)

# Only analyze if significant new data
if len(new_events) >= 10:
    run_full_analysis()
else:
    print("Not enough new data, skipping analysis")
```

---

## Multi-Project Analysis

Analyze patterns across multiple projects.

```bash
#!/bin/bash
# Analyze all projects in workspace

for project in ~/workspace/*/; do
    echo "Analyzing: $project"
    cd "$project"

    if [ -d ".claude/skills/soul" ]; then
        python .claude/skills/nexus/scripts/nexus_analyzer.py \
            --output "nexus_$(basename $project).md"
    fi
done

# Merge recommendations
python merge_nexus_reports.py ~/workspace/*/nexus_*.md
```

---

## Debugging and Troubleshooting

### Verbose Mode

```bash
python auto_skill_generator.py --verbose
```

Shows:
- All detected events
- Pattern matching details
- Priority calculations
- Merging logic
- Skill generation steps

### Debug Mode

```bash
NEXUS_DEBUG=1 python auto_skill_generator.py
```

Creates debug files:
- `.nexus_debug_events.json` - All SOUL events
- `.nexus_debug_patterns.json` - Detected patterns
- `.nexus_debug_merge.json` - Merging decisions

### Testing Pattern Detection

```python
#!/usr/bin/env python3
"""Test custom pattern matching"""
from nexus_analyzer import NEXUSUnifiedAnalyzer

analyzer = NEXUSUnifiedAnalyzer(threshold=1)  # Lower threshold for testing

# Add test events to SOUL
from soul_api import add_soul_event

add_soul_event("pdf_processing", "Test PDF extraction")
add_soul_event("pdf_processing", "Test PDF parsing")

# Run analysis
recs = analyzer.get_all_recommendations()

# Check if pattern detected
pdf_recs = [r for r in recs if r["pattern_type"] == "pdf_processing"]
print(f"PDF recommendations: {len(pdf_recs)}")
```

---

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) - Complete config reference
- [EXAMPLES.md](EXAMPLES.md) - Real-world examples
- [MANUAL_USAGE.md](MANUAL_USAGE.md) - Command-line options
- Main SKILL.md - NEXUS overview
