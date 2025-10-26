# PRD-TASKMASTER Configuration

## Task Detection Preferences

### Pattern Sensitivity
**High Sensitivity**: [ ] Detect all potential tasks including subtle patterns
**Balanced**: [ ] Standard task detection (recommended)
**Low Sensitivity**: [ ] Only detect clear, explicit tasks

### File Discovery
**Automatic**: [ ] Scan all markdown files for task patterns (recommended)
**Explicit Only**: [ ] Only analyze files with PRD/TODO/TASK in name
**Manual**: [ ] User specifies which files to analyze

## Skill Recommendation Settings

### Recommendation Threshold
**Low Threshold**: [ ] Recommend skill for 2+ similar tasks
**Medium Threshold**: [ ] Recommend skill for 4+ similar tasks (recommended)
**High Threshold**: [ ] Recommend skill for 6+ similar tasks

### Priority Calculation
**Task-Based**: [ ] Priority based solely on task count
**Impact-Based**: [ ] Consider task complexity and impact (recommended)
**Balanced**: [ ] Weight both task count and estimated impact

## Domain Classification

### Enable/Disable Domains
- [x] API Development
- [x] Testing & QA
- [x] Deployment & DevOps
- [x] Documentation
- [x] Database
- [x] Performance
- [x] Security
- [x] Data Processing
- [ ] Frontend Development
- [ ] Backend Development

### Custom Domain Keywords
Add custom keywords for domain detection:
```json
{
  "custom_domain_name": {
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "skill_name": "custom-skill-name"
  }
}
```

## Output Preferences

### Report Format
**Detailed**: [ ] Full analysis with all tasks and patterns (recommended)
**Summary**: [ ] Key findings and recommendations only
**Minimal**: [ ] Just skill recommendations

### Directive Generation
**Automatic**: [ ] Generate NEXUS directives automatically (recommended)
**Review First**: [ ] Show recommendations, wait for approval
**Manual**: [ ] Only analyze, don't generate directives

## Configuration File

Create `.prd_taskmaster_config.json`:

```json
{
  "pattern_sensitivity": "balanced",
  "file_discovery": "automatic",
  "recommendation_threshold": 4,
  "priority_calculation": "impact_based",
  "enabled_domains": [
    "api", "testing", "deployment", "documentation",
    "database", "performance", "security", "data_processing"
  ],
  "output_format": "detailed",
  "auto_generate_directives": true,
  "task_patterns": {
    "checkbox": true,
    "numbered": true,
    "bullet": true,
    "header": true
  }
}
```

## Advanced Options

### Task Complexity Analysis
**Enable**: [ ] Analyze task complexity for better prioritization
**Disable**: [ ] Simple frequency-based analysis (faster)

### Dependency Detection
**Enable**: [ ] Detect task dependencies and suggest skill order
**Disable**: [ ] Independent skill recommendations

### Integration with SOUL
**Enable**: [ ] Use SOUL data to inform recommendations (recommended)
**Disable**: [ ] Analyze current files only

## No Configuration Needed

**PRD-TASKMASTER works automatically without any configuration.**
These settings are optional optimizations for specific workflows.
