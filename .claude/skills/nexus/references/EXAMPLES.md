# NEXUS Real-World Examples

Real scenarios showing how NEXUS detects patterns and generates skills.

## Example 1: API Integration Skill

### Scenario

Developer working on a project that integrates with multiple external APIs.

### SOUL Events (Over 7 Days)

```
Day 1:
- api_call: Called GitHub API to fetch repos
- api_call: Called Stripe API for customer data
- error: GitHub API rate limit exceeded

Day 2:
- api_call: Called Stripe API for invoices
- api_call: Called SendGrid API for email
- error: Stripe API timeout

Day 3:
- api_call: Called GitHub API for issues
- api_call: Called Slack API for notifications
- api_call: Called Stripe API for subscriptions

... (continuing through Day 7)

Total: 24 API calls, 5 API errors
```

### NEXUS Analysis

```bash
python .claude/skills/nexus/scripts/auto_skill_generator.py
```

**Detected Pattern:**
- Pattern type: `api_call`
- Count: 24 occurrences
- Frequency: 3.4 times/day
- Priority: **CRITICAL** (frequency >= 2/day)

**Recommendation:**

```markdown
### 🔴 api-optimizer (CRITICAL)

**Pattern Type:** api_call
**Frequency:** 3.4 times/day (24 total)
**Reason:** Frequent API operations with error patterns detected

**Example Contexts:**
- Called GitHub API (8 times)
- Called Stripe API (10 times)
- Called SendGrid API (4 times)
- Called Slack API (2 times)

**Suggested Capabilities:**
- Rate limiting and backoff
- Retry logic for failed requests
- Response caching with TTL
- Error handling patterns
- Request logging and monitoring
```

**Auto-Generated Skill:**

NEXUS automatically creates `.claude/skills/api-optimizer/` with:
- SKILL.md with API best practices
- scripts/api_helper.py with retry/caching logic
- SOUL API integration to log API calls

### Result

Next time developer mentions "call API", Claude uses the `api-optimizer` skill automatically, applying rate limiting and retry logic.

---

## Example 2: Testing Skill from PRD

### Scenario

Project has extensive testing requirements in PRD.

### PRD File (PROJECT_PRD.md)

```markdown
## Testing Requirements

- [ ] Unit tests for auth module
- [ ] Integration tests for API endpoints
- [ ] End-to-end tests for checkout flow
- [ ] Performance tests for database queries
- [ ] Security tests for authentication
- [ ] Regression tests for bug fixes
- [ ] Test data generators
- [ ] Mock external services
- [ ] Code coverage > 80%
- [ ] Automated test runs in CI/CD
- [ ] Test documentation
- [ ] Visual regression tests
- [ ] Load testing for API
- [ ] Accessibility tests
- [ ] Cross-browser compatibility tests

Total: 15 testing tasks
```

### NEXUS Analysis

```bash
python .claude/skills/nexus/scripts/nexus_analyzer.py --prd-only
```

**Detected Pattern:**
- Pattern type: `testing`
- Task count: 15 tasks
- Source: PRD analysis
- Priority: **HIGH** (10-20 tasks)

**Recommendation:**

```markdown
### 🟠 test-guardian (HIGH)

**Pattern Type:** testing
**Source:** PRD analysis
**Task Count:** 15 test-related tasks

**PRD Files:**
- PROJECT_PRD.md (15 tasks)

**Suggested Capabilities:**
- Test case generation
- Mock data creation
- Coverage analysis
- Test runner integration
- CI/CD test automation
```

**Auto-Generated Skill:**

```bash
python .claude/skills/nexus/scripts/auto_skill_generator.py
```

Creates `.claude/skills/test-guardian/` with:
- SKILL.md with testing best practices
- scripts/test_generator.py for test scaffolding
- Coverage analysis tools
- Mock data generators

---

## Example 3: Combined Pattern (SOUL + PRD)

### Scenario

Data processing appears in both SOUL memory AND PRD requirements.

### SOUL Events

```
- data_processing: Parsed CSV file with user data (5 times)
- data_processing: Transformed JSON API response (6 times)
- data_processing: Validated data against schema (4 times)

Total: 15 data processing events over 7 days
Frequency: 2.1 times/day
```

### PRD Tasks

```markdown
## Data Pipeline Requirements

- [ ] ETL pipeline for user data
- [ ] CSV import functionality
- [ ] JSON transformation layer
- [ ] Data validation rules
- [ ] Schema evolution support
- [ ] Error handling for malformed data
- [ ] Bulk data operations

Total: 7 data processing tasks
```

### NEXUS Analysis

```bash
python .claude/skills/nexus/scripts/auto_skill_generator.py
```

**Pattern Merging:**

```
SOUL pattern: data_processing
  Count: 15
  Frequency: 2.1/day
  Individual priority: HIGH

PRD pattern: data_processing
  Task count: 7
  Individual priority: MEDIUM

MERGED priority: CRITICAL
  (SOUL HIGH + PRD MEDIUM = CRITICAL when combined)
```

**Recommendation:**

```markdown
### 🔴 data-transformer (CRITICAL)

**Pattern Type:** data_processing
**Source:** SOUL patterns + PRD analysis
**SOUL Frequency:** 2.1 times/day (15 total)
**PRD Tasks:** 7 ETL-related tasks

**Combined Reason:**
Pattern detected in both actual usage (SOUL) and planned requirements (PRD),
indicating critical need for dedicated data processing skill.

**Example Contexts (SOUL):**
- Parsed CSV files (5 times)
- Transformed JSON responses (6 times)
- Validated data schemas (4 times)

**Requirements (PRD):**
- ETL pipeline
- CSV/JSON handling
- Data validation
- Schema support

**Suggested Capabilities:**
- CSV/JSON/XML parsing
- Data validation and cleaning
- Format conversions
- Schema validation
- Bulk operations
- Error handling
```

**Result:**

Higher priority due to merging → Immediate auto-generation of `data-transformer` skill.

---

## Example 4: Error Pattern Detection

### Scenario

Recurring error pattern suggests need for error-handling skill.

### SOUL Events

```
Day 1:
- error: Database connection timeout
- error_resolved: Implemented retry logic

Day 2:
- error: External API unavailable
- error_resolved: Added fallback mechanism

Day 3:
- error: Database connection timeout (again)
- error: JSON parsing failed on malformed input

Day 4:
- error_resolved: Added connection pooling

... continuing

Total: 12 errors in 7 days
Frequency: 1.7 times/day
```

### NEXUS Analysis

```bash
python .claude/skills/nexus/scripts/auto_skill_generator.py --verbose
```

**Detected Pattern:**
- Pattern type: `error`
- Count: 12 occurrences
- Frequency: 1.7 times/day
- Priority: **HIGH** (frequency >= 1/day)

**Recommendation:**

```markdown
### 🟠 error-guardian (HIGH)

**Pattern Type:** error
**Frequency:** 1.7 times/day (12 total)
**Reason:** Recurring error patterns requiring systematic handling

**Example Error Types:**
- Database timeouts (4 times)
- API unavailability (3 times)
- JSON parsing errors (2 times)
- Network issues (3 times)

**Resolution Patterns Detected:**
- Retry logic
- Fallback mechanisms
- Connection pooling
- Input validation

**Suggested Capabilities:**
- Centralized error handling
- Retry strategies (exponential backoff)
- Circuit breaker pattern
- Fallback mechanisms
- Error logging and monitoring
- Input validation helpers
```

**Auto-Generated Skill:**

Creates error handling skill with patterns learned from previous resolutions.

---

## Example 5: No Generation (Pattern Below Threshold)

### Scenario

Pattern detected but below threshold.

### SOUL Events

```
- deployment: Deployed to staging (2 times)
- deployment: Updated production config (1 time)

Total: 3 deployment events in 7 days
Frequency: 0.4 times/day
```

### NEXUS Analysis

```bash
python .claude/skills/nexus/scripts/auto_skill_generator.py --dry-run
```

**Output:**

```
📊 Analysis Results:
   Total recommendations: 0

⚠️  Patterns below threshold (monitoring):
   - deployment: 3 occurrences (threshold: 5)

💡 These patterns are being monitored. If frequency increases,
   they will be recommended in future analyses.
```

**Result:**

No skill generated. Pattern monitored for future increase.

If deployment frequency increases to 5+ occurrences, NEXUS will automatically recommend `deploy-sage` skill.

---

## Example 6: Preventing Duplicates

### Scenario

Pattern detected but skill already exists.

### Existing Skills

```
.claude/skills/
├── api-optimizer/
├── test-guardian/
└── soul/
```

### SOUL Events

```
- api_call: Called GitHub API (occurred 20 times in 7 days)
Frequency: 2.9 times/day
Priority: CRITICAL
```

### NEXUS Analysis

```bash
python .claude/skills/nexus/scripts/auto_skill_generator.py
```

**Output:**

```
🔍 Analyzing patterns...
   Detected pattern: api_call (20 occurrences, CRITICAL)

🔍 Checking existing skills...
   ✓ api-optimizer already exists

⊗ Skipping api-optimizer (already exists)

✅ Analysis complete: 0 new skills generated
```

**Result:**

No duplicate created. Existing `api-optimizer` skill is sufficient.

---

## Example 7: Custom Threshold

### Scenario

Developer wants earlier skill generation (more aggressive).

### Command

```bash
# Lower threshold to 3 occurrences instead of 5
python .claude/skills/nexus/scripts/auto_skill_generator.py \
  --threshold 3 \
  --auto-threshold medium
```

### SOUL Events

```
- pdf_processing: Extracted text from PDF (4 times in 7 days)
Frequency: 0.6 times/day
```

**With Default Settings (threshold=5):**
- Pattern count: 4
- Threshold: 5
- Result: ❌ Not recommended (below threshold)

**With Custom Settings (threshold=3):**
- Pattern count: 4
- Threshold: 3
- Priority: MEDIUM (frequency 0.6/day)
- Auto-threshold: medium
- Result: ✅ `pdf-processor` skill auto-generated

---

## Example 8: Long-Term Analysis

### Scenario

Analyze patterns over longer period for strategic planning.

### Command

```bash
# Analyze last 30 days with higher threshold
python .claude/skills/nexus/scripts/nexus_analyzer.py \
  --days 30 \
  --threshold 15 \
  --output NEXUS_LONG_TERM.md
```

### Results

```markdown
# NEXUS Skill Recommendations (30-Day Analysis)

## Recommended Skills

### 1. 🔴 database-optimizer (CRITICAL)
**Count:** 67 database operations in 30 days
**Average:** 2.2 times/day
**Trend:** Increasing (week 1: 1.5/day → week 4: 3.0/day)

### 2. 🟠 security-auditor (HIGH)
**Count:** 42 security-related events in 30 days
**Average:** 1.4 times/day
**Trend:** Stable
```

Longer window reveals strategic needs not visible in 7-day window.

---

## Example 9: Multi-Source Workflow

### Scenario

Complete workflow showing all analysis sources.

### 1. Developer Works

```bash
# Developer uses Claude to build features
# SOUL automatically tracks:
- api_call: GitHub API (multiple times)
- data_processing: CSV imports (multiple times)
- error: Connection timeouts (few times)
```

### 2. PRD Exists

```markdown
## PROJECT_PRD.md

### Testing Requirements
- [ ] Unit tests
- [ ] Integration tests
... (12 total test tasks)

### API Integration
- [ ] GitHub OAuth
- [ ] Stripe payments
... (8 total API tasks)
```

### 3. TODOs Tracked

```markdown
## TODO.md

- [ ] Deploy to staging
- [ ] Set up production monitoring
- [ ] Configure CI/CD
... (6 deployment tasks)
```

### 4. NEXUS Analyzes (Automatic)

```bash
# Runs every 30 minutes via cron
*/30 * * * * /path/to/nexus_auto_watch.sh
```

### 5. Results

```
🔴 api-optimizer (CRITICAL)
   - SOUL: 24 api_calls
   - PRD: 8 API tasks
   - Combined priority: CRITICAL
   → AUTO-GENERATED ✅

🟠 test-guardian (HIGH)
   - PRD: 12 test tasks
   - Priority: HIGH
   → AUTO-GENERATED ✅

🟡 deploy-helper (MEDIUM)
   - TODO: 6 deployment tasks
   - Priority: MEDIUM
   → RECOMMENDED (not auto-generated)
```

### 6. Developer Benefits

Next session, developer has:
- `api-optimizer` skill ready for GitHub/Stripe integration
- `test-guardian` skill for automated testing
- Recommendation for `deploy-helper` to consider

All generated automatically based on actual work patterns!

---

## See Also

- [MANUAL_USAGE.md](MANUAL_USAGE.md) - Command-line options
- [OUTPUT_FORMAT.md](OUTPUT_FORMAT.md) - Output specification
- [ADVANCED.md](ADVANCED.md) - Advanced features
- Main SKILL.md - NEXUS overview
