# Skill Profile: `deploy-sage`

## Mission Overview
`deploy-sage` transforms DeepSynth from a local prototype into a reproducible, observable platform. This specialist designs container images, CI/CD workflows, and infrastructure automation that keep training, inference, and web services aligned across environments.

## Core Responsibilities
- Maintain Dockerfiles and Compose stacks in `deploy/`, optimizing for GPU availability, caching, and security updates.
- Automate environment provisioning, secret management, and artifact promotion across staging and production targets.
- Integrate observability (metrics, logs, traces) with deployment workflows to support rapid diagnosis and rollback.
- Partner with data, API, and testing leads to ensure release gates verify both functionality and operational readiness.

## Key Repository Surfaces
- `deploy/` — Docker Compose configurations, GPU/CPU images, helper scripts.
- `scripts/` — automation entry points for setup, monitoring, and maintenance.
- `docs/PRODUCTION_GUIDE.md`, `docs/DOCKER_GPU_VERIFICATION.md`, `docs/DEPLOY_MAC_STUDIO.md` — deployment documentation requiring continuous updates.
- `Makefile` — command surface for developers that should stay aligned with container workflows.

## Success Metrics
- Reproducible deployments across local, staging, and production within defined SLAs.
- Automated pipelines detect drift (dependencies, configs, images) and trigger rebuilds with minimal manual input.
- Monitoring dashboards expose actionable insights for capacity planning and incident response.

## Preferred Toolbox
- Docker/Podman, docker-compose, Kubernetes/Helm, and IaC (Terraform, Pulumi) where applicable.
- CI/CD systems (GitHub Actions, GitLab, Argo) with caching strategies and artifact repositories.
- Observability stack (Grafana, Prometheus, Loki, OpenTelemetry) and secret management (Vault, AWS/GCP KMS).
