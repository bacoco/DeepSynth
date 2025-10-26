#!/usr/bin/env python3
"""
Enhanced Skill Templates for NEXUS Directive Generator
Provides detailed, specific templates for each skill type with:
- Specific capabilities and activation triggers
- Framework-specific guidance
- File and code patterns
- Implementation phases
- Success metrics
"""

ENHANCED_SKILL_TEMPLATES = {
    "api": {
        "name": "API-MASTER",
        "title": "API Development Acceleration",
        "purpose": "Automate API development workflows including endpoint generation, schema validation, documentation, testing, and best practice enforcement for REST, GraphQL, and RPC APIs.",
        "detection_summary": "API development, integration, and optimization",
        "file_patterns": [
            "**/api/**/*.{py,js,ts,go,java}",
            "**/routes/**/*.{py,js,ts}",
            "**/controllers/**/*.{py,js,ts}",
            "**/graphql/**/*.{py,js,ts}",
            "**/*_controller.{py,rb}",
            "**/handlers/**/*.go"
        ],
        "code_patterns": [
            "@app.route(, @router.get(, @api.route(",
            "app.get(, app.post(, router.get(",
            "class.*APIView, class.*ViewSet",
            "type Query, type Mutation (GraphQL)",
            "@RestController, @RequestMapping (Java)"
        ],
        "keywords": ["API", "endpoint", "REST", "GraphQL", "RPC", "schema", "OpenAPI", "Swagger", "request validation", "API docs"],
        "capabilities": {
            "API Pattern Detection": [
                "Auto-detect REST/GraphQL/gRPC API files",
                "Identify API frameworks (FastAPI, Express, Flask, Django REST, etc.)",
                "Recognize API patterns in code (controllers, routes, handlers)"
            ],
            "Code Generation & Scaffolding": [
                "Generate REST endpoint boilerplate",
                "Create GraphQL resolvers and schemas",
                "Build API route handlers with proper structure",
                "Generate request/response models",
                "Create API client SDK code"
            ],
            "Schema Management": [
                "Validate request/response schemas",
                "Generate TypeScript/Python types from schemas",
                "Convert between schema formats (JSON Schema, OpenAPI, GraphQL SDL)",
                "Detect schema drift and breaking changes"
            ],
            "Documentation Automation": [
                "Generate OpenAPI/Swagger specs from code",
                "Create API documentation markdown",
                "Build interactive API explorers",
                "Maintain API changelog and versioning docs"
            ],
            "Best Practices Enforcement": [
                "Implement proper error handling patterns",
                "Add pagination to list endpoints",
                "Implement rate limiting",
                "Add request validation middleware",
                "Enforce consistent naming conventions",
                "Suggest caching strategies"
            ],
            "Testing Support": [
                "Generate integration tests for endpoints",
                "Create mock API responses",
                "Build test fixtures and factories",
                "Suggest edge cases for testing"
            ]
        },
        "frameworks": {
            "FastAPI": "Pydantic models, async/await, OpenAPI auto-generation",
            "Express.js": "Middleware patterns, error handling, routing",
            "Django REST": "ViewSets, serializers, permissions",
            "Flask": "Blueprints, decorators, request validation",
            "GraphQL": "Resolvers, type definitions, subscriptions"
        },
        "config_options": {
            "api_type": "rest | graphql | grpc | mixed",
            "framework": "auto | fastapi | express | django | flask | spring",
            "auth_method": "none | jwt | oauth2 | api_key | basic",
            "enable_auto_docs": "true | false",
            "enable_auto_validation": "true | false",
            "pagination_default": "offset | cursor | page"
        },
        "implementation_phases": [
            ("Core Detection", [
                "Implement file and code pattern detection",
                "Create basic API endpoint analyzer",
                "Build simple boilerplate generator"
            ]),
            ("Code Generation", [
                "Implement REST endpoint generator",
                "Add request/response model generation",
                "Create test generator"
            ]),
            ("Documentation", [
                "Build OpenAPI spec generator",
                "Add schema validation",
                "Implement breaking change detection"
            ]),
            ("Advanced Features", [
                "Add GraphQL support",
                "Implement client SDK generator",
                "Add performance optimization suggestions"
            ])
        ],
        "success_metrics": {
            "quantitative": [
                "Time to create new endpoint: < 5 minutes (vs 20+ minutes manual)",
                "API documentation coverage: > 95%",
                "Schema validation coverage: 100%",
                "Test coverage for API endpoints: > 80%"
            ],
            "qualitative": [
                "Consistent API design across all endpoints",
                "Reduced API-related bugs in production",
                "Faster API client integration",
                "Improved developer onboarding for API work"
            ]
        },
        "learning_resources": [
            "REST: RFC 7231 (HTTP semantics), Richardson Maturity Model",
            "GraphQL: GraphQL spec, Apollo best practices",
            "OpenAPI: OpenAPI 3.1 specification",
            "Security: OWASP API Security Top 10"
        ]
    },

    "testing": {
        "name": "TEST-GUARDIAN",
        "title": "Automated Testing & QA",
        "purpose": "Automate test generation, coverage analysis, and quality assurance workflows across unit, integration, and E2E tests.",
        "detection_summary": "testing, test coverage, and quality assurance",
        "file_patterns": [
            "**/tests/**/*.{py,js,ts,go,java}",
            "**/*_test.{py,js,ts,go}",
            "**/*.test.{js,ts,jsx,tsx}",
            "**/*.spec.{js,ts,jsx,tsx}",
            "**/test_*.py"
        ],
        "code_patterns": [
            "import pytest, import unittest",
            "describe(, it(, test(, expect(",
            "@Test, @pytest.mark",
            "func Test.* (Go)",
            "assert, assertEqual, toBe"
        ],
        "keywords": ["test", "testing", "pytest", "jest", "unittest", "coverage", "mock", "fixture"],
        "capabilities": {
            "Test Generation": [
                "Auto-generate unit tests from function signatures",
                "Create integration tests for APIs",
                "Generate E2E test scenarios",
                "Build test fixtures and factories"
            ],
            "Coverage Analysis": [
                "Identify untested code paths",
                "Calculate coverage gaps",
                "Suggest critical tests needed",
                "Track coverage trends"
            ],
            "Test Quality": [
                "Detect flaky tests",
                "Identify slow tests",
                "Suggest test improvements",
                "Enforce testing best practices"
            ],
            "Mocking & Fixtures": [
                "Generate mock objects",
                "Create test data factories",
                "Build reusable fixtures",
                "Handle external dependencies"
            ]
        },
        "frameworks": {
            "pytest": "Fixtures, parametrize, markers",
            "Jest": "Describe/it blocks, mocks, snapshots",
            "unittest": "TestCase classes, setUp/tearDown",
            "Mocha/Chai": "BDD style, assertions",
            "Go testing": "Table-driven tests, benchmarks"
        },
        "implementation_phases": [
            ("Core Generation", ["Basic test scaffolding", "Simple unit tests", "Test file structure"]),
            ("Advanced Testing", ["Integration tests", "Mock generation", "Fixtures/factories"]),
            ("Coverage & Quality", ["Coverage analysis", "Quality metrics", "Flaky test detection"]),
            ("Optimization", ["Parallel test running", "Test performance", "CI/CD integration"])
        ]
    },

    "deployment": {
        "name": "DEPLOY-SAGE",
        "title": "Deployment & Infrastructure Automation",
        "purpose": "Automate deployment workflows, container orchestration, CI/CD pipelines, and infrastructure as code.",
        "detection_summary": "deployment, CI/CD, and infrastructure automation",
        "file_patterns": [
            "Dockerfile", "docker-compose.yml",
            ".github/workflows/*.{yml,yaml}",
            ".gitlab-ci.yml",
            "**/*.tf (Terraform)",
            "**/k8s/**/*.{yml,yaml}",
            "**/*.yaml (Kubernetes)"
        ],
        "code_patterns": [
            "FROM, RUN, CMD (Docker)",
            "kubectl, helm",
            "terraform apply",
            "stages:, jobs:, steps: (CI/CD)"
        ],
        "keywords": ["deploy", "deployment", "docker", "kubernetes", "k8s", "CI/CD", "pipeline", "terraform"],
        "capabilities": {
            "Container Management": [
                "Optimize Dockerfile configurations",
                "Multi-stage build optimization",
                "Security scanning and hardening",
                "Container size reduction"
            ],
            "CI/CD Pipelines": [
                "Generate GitHub Actions workflows",
                "Create GitLab CI configurations",
                "Optimize pipeline performance",
                "Implement deployment strategies"
            ],
            "Kubernetes": [
                "Generate deployment manifests",
                "Create service configurations",
                "Implement health checks",
                "Set up auto-scaling"
            ],
            "Infrastructure as Code": [
                "Terraform module creation",
                "CloudFormation templates",
                "Resource optimization",
                "Cost analysis"
            ]
        },
        "frameworks": {
            "Docker": "Multi-stage builds, layer caching",
            "Kubernetes": "Deployments, Services, Ingress",
            "Terraform": "Modules, providers, state management",
            "GitHub Actions": "Workflows, jobs, steps, artifacts"
        }
    },

    "documentation": {
        "name": "DOC-GENIUS",
        "title": "Documentation Generation & Maintenance",
        "purpose": "Automate documentation generation, README creation, API docs, and keep documentation synchronized with code.",
        "detection_summary": "documentation, README, and API docs",
        "file_patterns": [
            "README.md", "**/*.md",
            "**/docs/**/*.{md,rst}",
            "**/*.rst",
            "mkdocs.yml", "docusaurus.config.js"
        ],
        "code_patterns": [
            "\"\"\" (docstrings)",
            "/** JSDoc */",
            "/// (C# docs)",
            "## (markdown headers)"
        ],
        "keywords": ["documentation", "README", "docs", "docstring", "markdown", "API docs"],
        "capabilities": {
            "README Generation": [
                "Generate comprehensive README files",
                "Create installation instructions",
                "Build usage examples",
                "Add badges and status indicators"
            ],
            "API Documentation": [
                "Generate API reference docs",
                "Create endpoint documentation",
                "Build interactive examples",
                "Maintain changelog"
            ],
            "Code Documentation": [
                "Generate docstrings from code",
                "Create inline documentation",
                "Build technical guides",
                "Document architecture decisions"
            ],
            "Maintenance": [
                "Detect outdated documentation",
                "Sync docs with code changes",
                "Validate links and examples",
                "Update version information"
            ]
        }
    },

    "performance": {
        "name": "PERF-OPTIMIZER",
        "title": "Performance Analysis & Optimization",
        "purpose": "Analyze performance bottlenecks, optimize code execution, implement caching strategies, and improve system efficiency.",
        "detection_summary": "performance optimization and bottleneck analysis",
        "file_patterns": [
            "**/*.py", "**/*.js", "**/*.go",
            "**/benchmark/**/*",
            "**/*_bench.go"
        ],
        "code_patterns": [
            "time.time(, performance.now(",
            "@profile, @benchmark",
            "SELECT *, WHERE (SQL queries)",
            "for .* in .* (loops)"
        ],
        "keywords": ["performance", "optimization", "cache", "speed", "profiling", "bottleneck"],
        "capabilities": {
            "Profiling & Analysis": [
                "Identify CPU bottlenecks",
                "Analyze memory usage",
                "Profile database queries",
                "Detect N+1 query problems"
            ],
            "Optimization": [
                "Optimize algorithm complexity",
                "Implement caching strategies",
                "Reduce memory allocation",
                "Parallelize workloads"
            ],
            "Database": [
                "Optimize SQL queries",
                "Add missing indexes",
                "Implement query caching",
                "Connection pool tuning"
            ],
            "Monitoring": [
                "Set up performance metrics",
                "Create alerting rules",
                "Build performance dashboards",
                "Track regression trends"
            ]
        }
    },

    "data_processing": {
        "name": "DATA-WIZARD",
        "title": "Data Processing & Transformation",
        "purpose": "Automate data processing pipelines, ETL workflows, data validation, and transformation tasks.",
        "detection_summary": "data processing, ETL, and transformation",
        "file_patterns": [
            "**/*.csv", "**/*.json", "**/*.xml",
            "**/etl/**/*.py",
            "**/data/**/*.py",
            "**/pipelines/**/*"
        ],
        "code_patterns": [
            "pandas.read_*, pd.DataFrame",
            "json.load, json.dumps",
            "csv.reader, csv.writer",
            "ETL, extract, transform, load"
        ],
        "keywords": ["data", "ETL", "pipeline", "transform", "csv", "json", "pandas"],
        "capabilities": {
            "Data Ingestion": [
                "Read multiple data formats",
                "Handle large datasets efficiently",
                "Stream processing support",
                "API data extraction"
            ],
            "Transformation": [
                "Clean and normalize data",
                "Handle missing values",
                "Type conversion and validation",
                "Data enrichment"
            ],
            "Validation": [
                "Schema validation",
                "Data quality checks",
                "Duplicate detection",
                "Anomaly identification"
            ],
            "Export": [
                "Multiple output formats",
                "Database loading",
                "API integration",
                "File system operations"
            ]
        }
    },

    "database": {
        "name": "DB-WIZARD",
        "title": "Database Management & Optimization",
        "purpose": "Automate database operations, query optimization, migration management, and schema design.",
        "detection_summary": "database operations, queries, and migrations",
        "file_patterns": [
            "**/migrations/**/*.{sql,py}",
            "**/*.sql",
            "**/models/**/*.{py,js,ts}",
            "**/schema/**/*.sql"
        ],
        "code_patterns": [
            "SELECT, INSERT, UPDATE, DELETE",
            "CREATE TABLE, ALTER TABLE",
            "db.query, session.execute",
            "Migration, migrate"
        ],
        "keywords": ["database", "sql", "query", "migration", "schema", "orm"],
        "capabilities": {
            "Query Optimization": [
                "Analyze slow queries",
                "Add missing indexes",
                "Optimize JOIN operations",
                "Suggest query rewrites"
            ],
            "Schema Management": [
                "Design database schemas",
                "Generate migrations",
                "Version control schemas",
                "Handle schema evolution"
            ],
            "ORM Integration": [
                "SQLAlchemy model generation",
                "TypeORM entities",
                "Mongoose schemas",
                "ActiveRecord models"
            ],
            "Data Integrity": [
                "Add constraints and validations",
                "Implement foreign keys",
                "Set up triggers",
                "Ensure referential integrity"
            ]
        }
    }
}


def get_enhanced_template(skill_type: str):
    """Get enhanced template for a skill type"""
    return ENHANCED_SKILL_TEMPLATES.get(skill_type, None)


def get_all_skill_types():
    """Get all available skill types"""
    return list(ENHANCED_SKILL_TEMPLATES.keys())
