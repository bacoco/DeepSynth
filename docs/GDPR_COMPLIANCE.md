# GDPR Compliance Guide

Complete guide to GDPR-compliant operation of DeepSynth OCR pipeline with built-in privacy controls.

## Table of Contents

1. [Overview](#overview)
2. [Privacy Controls](#privacy-controls)
3. [Configuration](#configuration)
4. [Data Processing](#data-processing)
5. [User Rights](#user-rights)
6. [Audit Logging](#audit-logging)
7. [Production Checklist](#production-checklist)

---

## Overview

DeepSynth includes built-in GDPR compliance features covering:

- **Article 4:** Pseudonymisation and anonymization
- **Article 5:** Data minimization and storage limitation
- **Article 7:** Consent management
- **Article 17:** Right to erasure (right to be forgotten)
- **Article 32:** Security of processing

### Compliance Architecture

```
┌──────────────────────────────────────────────────┐
│              User Request                        │
└───────────────┬──────────────────────────────────┘
                │
        ┌───────▼────────┐
        │ Consent Check  │
        │  (Article 7)   │
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  PII Redaction │
        │   (Article 5)  │
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │   Processing   │
        │ (Minimization) │
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  Data Storage  │
        │ (Retention)    │
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  Audit Log     │
        │  (Compliance)  │
        └────────────────┘
```

---

## Privacy Controls

### PrivacyConfig Class

All privacy settings are managed through the `PrivacyConfig` class:

```python
from deepsynth.config.env import get_config

config = get_config()

# Check privacy settings
print(f"Sample persistence: {config.privacy.allow_sample_persistence}")
print(f"PII redaction: {config.privacy.redact_pii_in_logs}")
print(f"Data retention: {config.privacy.data_retention_days} days")
print(f"Consent required: {config.privacy.require_consent}")
print(f"Metrics anonymization: {config.privacy.anonymize_metrics}")
```

### Privacy Settings

#### 1. Sample Persistence Control (Article 17)

Controls whether samples can be saved to disk:

```python
# Environment variable
ALLOW_SAMPLE_PERSISTENCE=false  # Default: false

# Check before saving
config = get_config()
if config.privacy.allow_sample_persistence:
    save_sample_to_disk(sample)
else:
    logger.info("Sample persistence disabled (GDPR Article 17)")
```

**Rationale:** Prevents unauthorized storage of user data, supporting "Right to Erasure."

#### 2. PII Redaction in Logs (Article 5)

Automatically redacts sensitive information from logs:

```python
# Environment variable
REDACT_PII_IN_LOGS=true  # Default: true

# Automatic redaction
logger.info(f"Processing password: {password}")
# Logs: "Processing password: <REDACTED_PASSWORD>"

logger.info(f"API token: {token}")
# Logs: "API token: <REDACTED_TOKEN>"

logger.info(f"User secret: {secret}")
# Logs: "User secret: <REDACTED_SECRET>"
```

**Redacted Keywords:**
- password
- token
- key
- secret
- api_key
- bearer
- authorization

**Rationale:** Data minimization principle - logs should not contain PII.

#### 3. Data Retention (Article 5)

Defines how long data is retained:

```python
# Environment variable
DATA_RETENTION_DAYS=90  # Default: 90 days

# Implementation
config = get_config()
retention_days = config.privacy.data_retention_days

# Delete old data
cutoff_date = datetime.now() - timedelta(days=retention_days)
delete_data_older_than(cutoff_date)
```

**Retention Policies:**
- **Development:** 30 days
- **Staging:** 60 days
- **Production:** 90 days (configurable)

**Rationale:** Storage limitation principle - data should not be kept longer than necessary.

#### 4. Consent Requirement (Article 7)

Requires explicit user consent:

```python
# Environment variable
REQUIRE_CONSENT=true  # Default: true (required in production)

# Check consent before processing
config = get_config()
if config.privacy.require_consent:
    if not user_has_consented(user_id):
        raise ConsentRequiredError("User consent required for processing")

process_data(data)
```

**Consent Management:**

```python
# Grant consent
def grant_consent(user_id: str, purpose: str) -> bool:
    """Grant user consent for specific purpose."""
    consent_record = {
        "user_id": user_id,
        "purpose": purpose,
        "granted_at": datetime.now(),
        "ip_address": get_client_ip(),  # Optional
    }
    store_consent(consent_record)
    return True

# Revoke consent
def revoke_consent(user_id: str, purpose: str) -> bool:
    """Revoke user consent."""
    delete_consent(user_id, purpose)
    delete_user_data(user_id)  # Right to erasure
    return True

# Check consent
def has_consent(user_id: str, purpose: str) -> bool:
    """Check if user has granted consent."""
    consent = get_consent(user_id, purpose)
    return consent is not None and consent.is_active
```

**Rationale:** Lawful processing requires explicit, informed consent.

#### 5. Metrics Anonymization (Article 4)

Anonymizes metrics and telemetry:

```python
# Environment variable
ANONYMIZE_METRICS=true  # Default: true

# Anonymize user identifiers
config = get_config()
if config.privacy.anonymize_metrics:
    user_id = anonymize_identifier(user_id)  # SHA-256 hash

record_metric("request_count", 1, {"user": user_id})
```

**Anonymization Methods:**
- **Hashing:** SHA-256 with salt
- **Aggregation:** Only report aggregate statistics
- **Sampling:** Random sampling for analysis

**Rationale:** Pseudonymisation reduces re-identification risk.

---

## Configuration

### Production Environment

```bash
# .env.production
ENVIRONMENT=production
SERVICE_NAME=deepsynth-api

# Privacy Controls (GDPR Compliance)
ALLOW_SAMPLE_PERSISTENCE=false       # Article 17: Right to erasure
REDACT_PII_IN_LOGS=true              # Article 5: Data minimization
DATA_RETENTION_DAYS=90               # Article 5: Storage limitation
REQUIRE_CONSENT=true                 # Article 7: Consent
ANONYMIZE_METRICS=true               # Article 4: Pseudonymisation

# Security
ENABLE_AUDIT_LOG=true                # Track all data access
ENABLE_ENCRYPTION_AT_REST=true       # Article 32: Security
ENABLE_TLS=true                      # Article 32: Data in transit
```

### Development Environment

```bash
# .env.development
ENVIRONMENT=development
SERVICE_NAME=deepsynth-dev

# Privacy Controls (Relaxed for development)
ALLOW_SAMPLE_PERSISTENCE=true        # OK for local testing
REDACT_PII_IN_LOGS=true              # Still redact PII
DATA_RETENTION_DAYS=30               # Shorter retention
REQUIRE_CONSENT=false                # Not required for testing
ANONYMIZE_METRICS=false              # OK for debugging

# Security (Relaxed)
ENABLE_AUDIT_LOG=false
```

### Validation

Configuration is automatically validated on startup:

```python
from deepsynth.config.env import get_config

config = get_config()

# Production safety checks
if config.is_production:
    if not config.privacy.redact_pii_in_logs:
        raise ValueError("PII redaction required in production")

    if not config.privacy.require_consent:
        raise ValueError("Consent required in production")

    if config.privacy.allow_sample_persistence:
        logger.warning("Sample persistence enabled in production - ensure compliance")
```

---

## Data Processing

### Processing Principles

#### 1. Data Minimization

Only collect and process necessary data:

```python
# ❌ Bad: Collecting unnecessary data
data = {
    "image": image,
    "user_name": user_name,        # Not needed
    "email": email,                 # Not needed
    "ip_address": ip_address,       # Not needed
    "browser": browser,             # Not needed
}

# ✅ Good: Minimal data collection
data = {
    "image": image,                 # Necessary for OCR
    "request_id": generate_uuid(),  # For tracking, no PII
}
```

#### 2. Purpose Limitation

Process data only for specified purpose:

```python
def process_ocr_request(image, consent):
    """Process OCR request with consent validation."""

    # Check consent for specific purpose
    if not consent.has_purpose("ocr_processing"):
        raise ConsentError("User has not consented to OCR processing")

    # Process only for consented purpose
    result = ocr_service.infer(image)

    # Do NOT use for other purposes (e.g., training) without separate consent
    if consent.has_purpose("model_training"):
        add_to_training_dataset(image, result)

    return result
```

#### 3. Storage Limitation

Delete data when no longer needed:

```python
from datetime import datetime, timedelta

def cleanup_old_data():
    """Delete data older than retention period."""

    config = get_config()
    retention_days = config.privacy.data_retention_days

    cutoff_date = datetime.now() - timedelta(days=retention_days)

    # Delete old samples
    deleted_samples = delete_samples_before(cutoff_date)
    logger.info(f"Deleted {deleted_samples} old samples (GDPR Article 5)")

    # Delete old logs
    deleted_logs = delete_logs_before(cutoff_date)
    logger.info(f"Deleted {deleted_logs} old log entries")

    # Delete old audit records (keep longer for compliance)
    audit_retention_days = retention_days * 2
    audit_cutoff = datetime.now() - timedelta(days=audit_retention_days)
    deleted_audits = delete_audits_before(audit_cutoff)
    logger.info(f"Deleted {deleted_audits} old audit records")

# Schedule cleanup
import schedule
schedule.every().day.at("02:00").do(cleanup_old_data)
```

---

## User Rights

### Right to Access (Article 15)

Users can request their data:

```python
def get_user_data(user_id: str) -> dict:
    """Get all data associated with user."""

    data = {
        "user_id": user_id,
        "consents": get_user_consents(user_id),
        "processing_history": get_processing_history(user_id),
        "stored_samples": get_stored_samples(user_id),
        "audit_trail": get_audit_trail(user_id),
    }

    return data
```

### Right to Erasure (Article 17)

Users can request deletion:

```python
def delete_user_data(user_id: str) -> dict:
    """Delete all user data (right to be forgotten)."""

    logger.info(f"Processing deletion request for user: {user_id}")

    # Delete all user data
    results = {
        "consents_deleted": delete_consents(user_id),
        "samples_deleted": delete_samples(user_id),
        "history_deleted": delete_history(user_id),
        "cache_cleared": clear_user_cache(user_id),
    }

    # Audit deletion
    audit_log.record({
        "event": "user_data_deletion",
        "user_id": user_id,
        "timestamp": datetime.now(),
        "results": results,
    })

    logger.info(f"User data deleted: {user_id}")

    return results
```

### Right to Data Portability (Article 20)

Export user data in machine-readable format:

```python
def export_user_data(user_id: str, format: str = "json") -> bytes:
    """Export user data in requested format."""

    data = get_user_data(user_id)

    if format == "json":
        return json.dumps(data, indent=2).encode()
    elif format == "csv":
        return convert_to_csv(data)
    elif format == "xml":
        return convert_to_xml(data)
    else:
        raise ValueError(f"Unsupported format: {format}")
```

### Right to Rectification (Article 16)

Users can correct their data:

```python
def update_user_data(user_id: str, updates: dict) -> bool:
    """Update user data with corrections."""

    # Validate updates
    validate_updates(updates)

    # Apply updates
    for key, value in updates.items():
        update_field(user_id, key, value)

    # Audit update
    audit_log.record({
        "event": "user_data_updated",
        "user_id": user_id,
        "fields": list(updates.keys()),
        "timestamp": datetime.now(),
    })

    return True
```

---

## Audit Logging

### Audit Trail

Track all data access and processing:

```python
from deepsynth.utils.audit import audit_log

# Record processing event
audit_log.record({
    "event": "data_processed",
    "user_id": user_id,
    "purpose": "ocr_processing",
    "consent_verified": True,
    "data_type": "image",
    "timestamp": datetime.now(),
    "ip_address": request.remote_addr,
})

# Record data access
audit_log.record({
    "event": "data_accessed",
    "user_id": user_id,
    "accessor": "admin@company.com",
    "purpose": "support_request",
    "timestamp": datetime.now(),
})

# Record deletion
audit_log.record({
    "event": "data_deleted",
    "user_id": user_id,
    "reason": "user_request",
    "timestamp": datetime.now(),
})
```

### Audit Log Retention

Audit logs are retained longer than operational data:

```python
# Operational data: 90 days
DATA_RETENTION_DAYS=90

# Audit logs: 180 days (2x operational)
AUDIT_RETENTION_DAYS=180

# Legal hold: Indefinite (for compliance)
LEGAL_HOLD_ENABLED=true
```

---

## Production Checklist

### Pre-Deployment Checklist

- [ ] **Environment Configuration**
  - [ ] `ENVIRONMENT=production`
  - [ ] `REDACT_PII_IN_LOGS=true`
  - [ ] `REQUIRE_CONSENT=true`
  - [ ] `ANONYMIZE_METRICS=true`
  - [ ] `DATA_RETENTION_DAYS` set appropriately

- [ ] **Privacy Controls**
  - [ ] Sample persistence disabled or justified
  - [ ] PII redaction verified in logs
  - [ ] Consent management implemented
  - [ ] Data retention policy enforced

- [ ] **User Rights**
  - [ ] Right to access implemented
  - [ ] Right to erasure implemented
  - [ ] Right to portability implemented
  - [ ] Right to rectification implemented

- [ ] **Security**
  - [ ] TLS/HTTPS enabled
  - [ ] Encryption at rest enabled
  - [ ] Access controls configured
  - [ ] Audit logging enabled

- [ ] **Documentation**
  - [ ] Privacy policy published
  - [ ] Cookie policy (if applicable)
  - [ ] Terms of service updated
  - [ ] DPA (Data Processing Agreement) signed

- [ ] **Testing**
  - [ ] Privacy controls tested
  - [ ] Data deletion tested
  - [ ] Consent flow tested
  - [ ] Export functionality tested

### Compliance Verification

```bash
# Run compliance checks
python scripts/verify_compliance.py

# Expected output:
# ✅ PII redaction enabled
# ✅ Consent required in production
# ✅ Data retention policy configured
# ✅ Metrics anonymization enabled
# ✅ Audit logging enabled
# ✅ All checks passed
```

### Regular Audits

Schedule regular compliance audits:

```python
# Monthly compliance report
def generate_compliance_report():
    report = {
        "period": "2025-01",
        "data_processed": count_processed_records(),
        "consents_granted": count_consents(),
        "consents_revoked": count_revocations(),
        "deletion_requests": count_deletions(),
        "data_breaches": count_breaches(),  # Should be 0
        "audit_findings": get_audit_findings(),
    }

    save_report(report)
    notify_dpo(report)  # Data Protection Officer

# Schedule monthly
schedule.every().month.at("01:00").do(generate_compliance_report)
```

---

## Additional Resources

- **GDPR Official Text:** https://gdpr-info.eu/
- **ICO Guidance:** https://ico.org.uk/for-organisations/guide-to-data-protection/
- **EDPB Guidelines:** https://edpb.europa.eu/our-work-tools/general-guidance_en

---

## Questions & Support

For GDPR compliance questions:
- **Email:** dpo@yourdomain.com (Data Protection Officer)
- **Documentation:** See [deepseek_ocr_pipeline.md](./deepseek_ocr_pipeline.md)
- **Issues:** https://github.com/yourusername/DeepSynth/issues

---

**Important:** This guide provides technical implementation of GDPR controls. Consult with legal counsel for compliance verification in your specific jurisdiction.
