# NEXUS Installation Guide

Complete setup instructions for NEXUS automatic skill generation.

## Prerequisites

- Python 3.7 or higher
- SOUL skill installed
- Git repository (for automatic monitoring)

## Quick Install

```bash
# 1. NEXUS is already installed if you have .claude/skills/nexus/
cd /path/to/your/project

# 2. Verify NEXUS is present
ls -la .claude/skills/nexus/

# 3. Test NEXUS
python .claude/skills/nexus/scripts/nexus_analyzer.py --dry-run
```

That's it! NEXUS is ready to use.

## Automatic Monitoring Setup

For automatic skill generation, set up monitoring via cron or git hooks.

### Option 1: Cron Job (Recommended)

Automatically run NEXUS every 30 minutes:

```bash
# Edit crontab
crontab -e

# Add this line for every 30 minutes
*/30 * * * * /path/to/.claude/skills/scripts/nexus_auto_watch.sh >> /path/to/.nexus_cron.log 2>&1

# Or for hourly monitoring
0 * * * * /path/to/.claude/skills/scripts/nexus_auto_watch.sh >> /path/to/.nexus_cron.log 2>&1
```

**Adjust path:**

```bash
# Get absolute path
PROJECT_DIR=$(pwd)
echo "*/30 * * * * $PROJECT_DIR/.claude/skills/scripts/nexus_auto_watch.sh >> $PROJECT_DIR/.nexus_cron.log 2>&1" | crontab -
```

**Verify cron:**

```bash
# Check cron is scheduled
crontab -l

# Check logs after 30 minutes
tail -f .nexus_cron.log
```

### Option 2: Git Hook

Run NEXUS automatically after every commit:

```bash
# Create post-commit hook
cat > .git/hooks/post-commit << 'EOF'
#!/bin/bash
# Run NEXUS in background after commits

PROJECT_DIR=$(git rev-parse --show-toplevel)
"$PROJECT_DIR/.claude/skills/scripts/nexus_auto_watch.sh" >> "$PROJECT_DIR/.nexus_git_hook.log" 2>&1 &
EOF

# Make executable
chmod +x .git/hooks/post-commit
```

**Test git hook:**

```bash
# Make a test commit
echo "test" >> .gitignore
git add .gitignore
git commit -m "test: NEXUS git hook"

# Check hook ran
tail .nexus_git_hook.log
```

### Option 3: Manual Runs

Run NEXUS manually whenever needed:

```bash
# Analyze and auto-generate
python .claude/skills/nexus/scripts/auto_skill_generator.py

# Analyze only (no generation)
python .claude/skills/nexus/scripts/nexus_analyzer.py
```

## Configuration Setup

### Basic Configuration

Create `.nexus_config.json`:

```bash
cat > .nexus_config.json << 'EOF'
{
  "analysis": {
    "threshold": 5,
    "window_days": 7,
    "auto_threshold": "high"
  },
  "sources": {
    "soul_memory": true,
    "prd_files": true,
    "task_lists": true
  },
  "auto_generation": {
    "enabled": true,
    "max_skills_per_run": 5,
    "log_to_soul": true
  },
  "logging": {
    "enabled": true,
    "file": ".nexus_auto.log",
    "level": "INFO"
  }
}
EOF
```

### Project-Specific Paths

For non-standard project structures:

```json
{
  "paths": {
    "skills_dir": ".claude/skills",
    "prd_patterns": ["docs/*PRD*.md", "requirements/*.md"],
    "todo_patterns": ["tasks/*.md", "TODO.md"]
  }
}
```

## Verification

### Test NEXUS Installation

```bash
# 1. Check NEXUS files exist
ls -la .claude/skills/nexus/scripts/

# Expected output:
# nexus_analyzer.py
# auto_skill_generator.py
# soul_integration.py

# 2. Test analyzer
python .claude/skills/nexus/scripts/nexus_analyzer.py --dry-run

# Expected output:
# 📊 NEXUS Unified Analyzer
# ✅ Analysis complete

# 3. Test auto-generator
python .claude/skills/nexus/scripts/auto_skill_generator.py --dry-run

# Expected output:
# 🤖 NEXUS Auto Skill Generator (DRY RUN)
# 📊 Analysis Results: ...
```

### Test SOUL Integration

```bash
# 1. Check SOUL is available
python -c "from soul_api import get_soul_memory; print('✅ SOUL API available')"

# 2. Add test event
python -c "from soul_api import add_soul_event; add_soul_event('test', 'NEXUS test event')"

# 3. Run NEXUS to detect
python .claude/skills/nexus/scripts/nexus_analyzer.py
```

### Test Automatic Monitoring

If using cron:

```bash
# Wait 30 minutes, then check logs
tail -f .nexus_cron.log
```

If using git hook:

```bash
# Make a test commit
git commit --allow-empty -m "test: NEXUS monitoring"

# Check log
tail .nexus_git_hook.log
```

## Troubleshooting Installation

### SOUL Not Found

**Error:**
```
ModuleNotFoundError: No module named 'soul_api'
```

**Solution:**

```bash
# Check SOUL is installed
ls -la .claude/skills/soul/scripts/soul_api.py

# Install SOUL if missing
cd .claude/skills/soul/scripts
./install.sh
```

### Permission Denied

**Error:**
```
Permission denied: .nexus_auto.log
```

**Solution:**

```bash
# Create log file with correct permissions
touch .nexus_auto.log
chmod 644 .nexus_auto.log

# For cron, ensure directory is writable
chmod 755 $(pwd)
```

### Cron Not Running

**Error:**
Cron job not executing

**Solution:**

```bash
# Check cron service is running
systemctl status cron   # Linux
sudo launchctl list | grep cron  # macOS

# Check cron logs
grep NEXUS /var/log/syslog  # Linux
grep NEXUS /var/log/system.log  # macOS

# Use absolute paths in crontab
crontab -e
# Change: */30 * * * * ./script.sh
# To: */30 * * * * /full/path/to/script.sh
```

### Git Hook Not Triggering

**Error:**
Hook exists but doesn't run

**Solution:**

```bash
# Check hook is executable
ls -la .git/hooks/post-commit
# Should show: -rwxr-xr-x

# Make executable if needed
chmod +x .git/hooks/post-commit

# Test hook manually
.git/hooks/post-commit

# Check for errors
cat .nexus_git_hook.log
```

## Uninstallation

### Remove Cron Job

```bash
# Edit crontab
crontab -e

# Remove NEXUS line, save and exit

# Verify removal
crontab -l | grep nexus
# Should show nothing
```

### Remove Git Hook

```bash
rm .git/hooks/post-commit
```

### Remove Configuration

```bash
rm .nexus_config.json
rm .nexus_auto.log
rm .nexus_cron.log
rm .nexus_git_hook.log
```

### Keep NEXUS, Disable Auto-Generation

```json
{
  "auto_generation": {
    "enabled": false
  }
}
```

Or remove cron/hooks but keep NEXUS for manual use.

## Upgrading NEXUS

If NEXUS is updated:

```bash
# Pull latest version
git pull

# No additional steps needed - Python scripts auto-reload

# Test new version
python .claude/skills/nexus/scripts/auto_skill_generator.py --version
```

## Multi-Project Setup

### Shared NEXUS Installation

For multiple projects sharing skills:

```bash
# Project structure
~/workspace/
├── .shared_skills/
│   └── nexus/
└── project1/
    └── .claude/
        └── skills/ -> ../../.shared_skills/
```

### Per-Project NEXUS

Each project has its own NEXUS:

```bash
~/workspace/
├── project1/
│   └── .claude/skills/nexus/
└── project2/
    └── .claude/skills/nexus/
```

Setup cron for each:

```bash
# Edit crontab
crontab -e

# Add line for each project
*/30 * * * * /home/user/workspace/project1/.claude/skills/scripts/nexus_auto_watch.sh
*/30 * * * * /home/user/workspace/project2/.claude/skills/scripts/nexus_auto_watch.sh
```

## Platform-Specific Notes

### Linux

- Cron service usually pre-installed
- Use `systemctl` to manage services
- Logs typically in `/var/log/syslog`

### macOS

- Use `cron` or `launchd` for scheduling
- `launchd` is preferred for system services
- Logs in `/var/log/system.log`

### Windows (WSL)

- Install cron: `sudo apt-get install cron`
- Start cron: `sudo service cron start`
- Use Linux instructions

### Windows (Native)

Use Task Scheduler instead of cron:

```powershell
# Create scheduled task
$action = New-ScheduledTaskAction -Execute "python" -Argument "C:\path\to\auto_skill_generator.py"
$trigger = New-ScheduledTaskTrigger -Once -At 12:00 -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Days 365)
Register-ScheduledTask -TaskName "NEXUS Monitor" -Action $action -Trigger $trigger
```

## Docker Setup

For containerized environments:

```dockerfile
# Dockerfile
FROM python:3.9

WORKDIR /app
COPY .claude/skills/nexus /app/.claude/skills/nexus

# Install cron
RUN apt-get update && apt-get install -y cron

# Add cron job
RUN echo "*/30 * * * * python /app/.claude/skills/nexus/scripts/auto_skill_generator.py" | crontab -

# Run cron in foreground
CMD ["cron", "-f"]
```

## See Also

- [MANUAL_USAGE.md](MANUAL_USAGE.md) - Command-line options
- [CONFIGURATION.md](CONFIGURATION.md) - Config reference
- [EXAMPLES.md](EXAMPLES.md) - Real-world examples
- Main SKILL.md - NEXUS overview
