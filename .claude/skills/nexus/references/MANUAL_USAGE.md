# NEXUS Manual Usage

Complete command-line reference for NEXUS tools.

## Auto-Generator (Recommended)

Analyzes patterns and automatically generates skills.

### Basic Usage

```bash
# Run with defaults (threshold=5, days=7, auto-threshold=high)
python .claude/skills/nexus/scripts/auto_skill_generator.py
```

This will:
1. Analyze SOUL memory for patterns
2. Analyze PRD files for requirements
3. Analyze task lists for TODOs
4. Generate skills for HIGH and CRITICAL priorities
5. Log generated skills in SOUL memory

### Command-Line Options

```bash
python .claude/skills/nexus/scripts/auto_skill_generator.py \
  --threshold 3 \           # Pattern occurrence threshold (default: 5)
  --days 14 \               # Analysis window in days (default: 7)
  --auto-threshold medium \ # Auto-generate if priority >= this (default: high)
  --dry-run \               # Analyze only, don't generate skills
  --output custom.md        # Custom output file (default: NEXUS_RECOMMENDATIONS.md)
```

### Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--threshold` | int | 5 | Minimum pattern occurrences to recommend |
| `--days` | int | 7 | Number of days to analyze |
| `--auto-threshold` | str | high | Auto-generate if priority >= this (low/medium/high/critical) |
| `--dry-run` | flag | false | Analyze only, don't create skills |
| `--output` | str | NEXUS_RECOMMENDATIONS.md | Output file path |
| `--config` | str | .nexus_config.json | Config file path |
| `--verbose` | flag | false | Detailed output |
| `--quiet` | flag | false | Minimal output |

### Examples

#### Dry Run (Analysis Only)

```bash
# See what would be generated without creating skills
python .claude/skills/nexus/scripts/auto_skill_generator.py --dry-run
```

Output:
```
🤖 NEXUS Auto Skill Generator (DRY RUN MODE)

📊 Analysis Results:
   Total recommendations: 3
   High priority: 2
   Medium priority: 1

🔧 Would generate (not actually creating):
   ✓ api-optimizer (priority: critical)
   ✓ data-transformer (priority: high)
   ⊗ deploy-helper (priority: medium - below threshold)

💾 Recommendations saved to: NEXUS_RECOMMENDATIONS.md
```

#### Lower Threshold

```bash
# More sensitive - generate skills for patterns >= 3 occurrences
python .claude/skills/nexus/scripts/auto_skill_generator.py --threshold 3
```

#### Longer Analysis Window

```bash
# Analyze last 30 days instead of 7
python .claude/skills/nexus/scripts/auto_skill_generator.py --days 30
```

#### Generate MEDIUM Priority Skills

```bash
# Auto-generate for medium and higher (more aggressive)
python .claude/skills/nexus/scripts/auto_skill_generator.py --auto-threshold medium
```

#### Conservative Mode

```bash
# Only generate CRITICAL skills
python .claude/skills/nexus/scripts/auto_skill_generator.py --auto-threshold critical
```

#### Verbose Output

```bash
# Show detailed analysis
python .claude/skills/nexus/scripts/auto_skill_generator.py --verbose
```

Output:
```
🤖 NEXUS Auto Skill Generator

🔍 Analyzing SOUL memory...
   Found 47 events in last 7 days
   Detected patterns:
     - api_call: 24 occurrences (3.4/day)
     - data_processing: 15 occurrences (2.1/day)
     - error: 8 occurrences (1.1/day)

🔍 Analyzing PRD files...
   Found 2 PRD files:
     - PROJECT_PRD.md (18 tasks)
     - TEST_REQUIREMENTS.md (7 tasks)

   Task breakdown:
     - testing: 15 tasks
     - api: 7 tasks
     - deployment: 3 tasks

🔍 Analyzing task lists...
   Found 1 TODO file:
     - TODO.md (12 tasks)

   Task types:
     - deployment: 5 tasks
     - testing: 4 tasks
     - docs: 3 tasks

🧮 Merging and prioritizing...
   Merged 'api' patterns (SOUL + PRD) → CRITICAL
   Merged 'testing' patterns (PRD + TODO) → HIGH

📊 Final recommendations: 3
   🔴 api-optimizer (critical)
   🟠 test-guardian (high)
   🟡 deploy-helper (medium)

🔧 Generating skills (auto-threshold: high)...
   ✅ api-optimizer created successfully
   ✅ test-guardian created successfully
   ⊗ deploy-helper skipped (below threshold)

✅ Auto-generation complete!
   Skills created: 2
   Skills skipped: 1
```

---

## Analyzer (Analysis Only)

Generates recommendations without auto-creating skills.

### Basic Usage

```bash
# Analyze and create NEXUS_RECOMMENDATIONS.md
python .claude/skills/nexus/scripts/nexus_analyzer.py
```

### Command-Line Options

```bash
python .claude/skills/nexus/scripts/nexus_analyzer.py \
  --threshold 3 \
  --days 14 \
  --output MY_RECOMMENDATIONS.md \
  --format markdown \
  --config .nexus_config.json
```

### Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--threshold` | int | 5 | Minimum pattern occurrences |
| `--days` | int | 7 | Analysis window in days |
| `--output` | str | NEXUS_RECOMMENDATIONS.md | Output file path |
| `--format` | str | markdown | Output format (markdown/json) |
| `--config` | str | .nexus_config.json | Config file path |
| `--soul-only` | flag | false | Only analyze SOUL patterns |
| `--prd-only` | flag | false | Only analyze PRD files |
| `--tasks-only` | flag | false | Only analyze task lists |

### Examples

#### JSON Output

```bash
# Generate machine-readable JSON
python .claude/skills/nexus/scripts/nexus_analyzer.py --format json --output recs.json
```

#### SOUL Analysis Only

```bash
# Only analyze SOUL memory patterns
python .claude/skills/nexus/scripts/nexus_analyzer.py --soul-only
```

#### PRD Analysis Only

```bash
# Only analyze PRD files
python .claude/skills/nexus/scripts/nexus_analyzer.py --prd-only
```

#### Multiple Analysis Runs

```bash
# Short-term patterns (last 3 days, lower threshold)
python .claude/skills/nexus/scripts/nexus_analyzer.py \
  --days 3 \
  --threshold 2 \
  --output NEXUS_SHORT_TERM.md

# Long-term patterns (last 30 days, higher threshold)
python .claude/skills/nexus/scripts/nexus_analyzer.py \
  --days 30 \
  --threshold 10 \
  --output NEXUS_LONG_TERM.md
```

---

## Monitoring Script

Wrapper for automatic periodic execution.

### Usage

```bash
# Run monitoring (calls auto_skill_generator.py)
/path/to/.claude/skills/scripts/nexus_auto_watch.sh
```

### Cron Setup

```bash
# Edit crontab
crontab -e

# Add one of these lines:

# Every 30 minutes
*/30 * * * * /path/to/.claude/skills/scripts/nexus_auto_watch.sh

# Every hour
0 * * * * /path/to/.claude/skills/scripts/nexus_auto_watch.sh

# Twice daily (9am and 5pm)
0 9,17 * * * /path/to/.claude/skills/scripts/nexus_auto_watch.sh

# Daily at midnight
0 0 * * * /path/to/.claude/skills/scripts/nexus_auto_watch.sh
```

### Git Hook Setup

```bash
# Add to .git/hooks/post-commit
cat >> .git/hooks/post-commit << 'EOF'
#!/bin/bash
# Run NEXUS in background after commits
/path/to/.claude/skills/scripts/nexus_auto_watch.sh &
EOF

chmod +x .git/hooks/post-commit
```

---

## Configuration File

Create `.nexus_config.json` for persistent settings:

```json
{
  "analysis": {
    "threshold": 5,
    "window_days": 7,
    "auto_threshold": "high"
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
  },
  "auto_generation": {
    "enabled": true,
    "max_skills_per_run": 5,
    "log_to_soul": true
  }
}
```

Then run without arguments:
```bash
# Uses config file settings
python .claude/skills/nexus/scripts/auto_skill_generator.py
```

Command-line args override config file.

---

## Return Codes

NEXUS scripts use standard exit codes:

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | SOUL not available |
| 4 | No patterns detected |
| 5 | Skill generation failed |

### Example Usage in Scripts

```bash
#!/bin/bash

python .claude/skills/nexus/scripts/auto_skill_generator.py
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ NEXUS ran successfully"
elif [ $exit_code -eq 4 ]; then
    echo "ℹ️  No patterns detected - nothing to do"
else
    echo "❌ NEXUS failed with code $exit_code"
fi
```

---

## Logs

NEXUS logs to `.nexus_auto.log`:

```bash
# View recent logs
tail -n 50 .nexus_auto.log

# Watch logs in real-time
tail -f .nexus_auto.log

# Search for errors
grep ERROR .nexus_auto.log

# Find generated skills
grep "created successfully" .nexus_auto.log
```

---

## Troubleshooting

### No Patterns Detected

```bash
# Lower threshold to detect more patterns
python .claude/skills/nexus/scripts/auto_skill_generator.py --threshold 2
```

### SOUL Not Available

```bash
# Check SOUL installation
python .claude/skills/soul/scripts/soul_api.py

# Reinstall if needed
cd .claude/skills/soul/scripts && ./install.sh
```

### Skills Not Generating

```bash
# Check with dry-run to see what would happen
python .claude/skills/nexus/scripts/auto_skill_generator.py --dry-run --verbose

# Try lower auto-threshold
python .claude/skills/nexus/scripts/auto_skill_generator.py --auto-threshold medium
```

### Permission Denied

```bash
# Make scripts executable
chmod +x .claude/skills/nexus/scripts/*.py
chmod +x .claude/skills/scripts/*.sh
```

---

## Advanced Usage

### Custom Analysis Pipeline

```python
#!/usr/bin/env python3
"""Custom NEXUS pipeline"""
import sys
from pathlib import Path

# Add NEXUS to path
nexus_scripts = Path(".claude/skills/nexus/scripts")
sys.path.insert(0, str(nexus_scripts))

from nexus_analyzer import NEXUSUnifiedAnalyzer
from auto_skill_generator import AutoSkillGenerator

# Custom analyzer
analyzer = NEXUSUnifiedAnalyzer(threshold=3, window_days=14)
recommendations = analyzer.get_all_recommendations()

# Filter for specific pattern types
api_recs = [r for r in recommendations if r['pattern_type'] == 'api_call']

# Custom generation logic
if len(api_recs) > 0:
    generator = AutoSkillGenerator(auto_threshold="medium")
    for rec in api_recs:
        generator.generate_skill_with_creator(rec)
```

### Integration with CI/CD

```yaml
# .github/workflows/nexus.yml
name: NEXUS Analysis

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run NEXUS
        run: |
          python .claude/skills/nexus/scripts/auto_skill_generator.py --dry-run
      - name: Upload recommendations
        uses: actions/upload-artifact@v2
        with:
          name: nexus-recommendations
          path: NEXUS_RECOMMENDATIONS.md
```

---

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) - Complete config reference
- [INSTALLATION.md](INSTALLATION.md) - Setup guide
- [EXAMPLES.md](EXAMPLES.md) - Real-world examples
- Main SKILL.md - NEXUS overview
