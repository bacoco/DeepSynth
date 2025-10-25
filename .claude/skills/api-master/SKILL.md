---
name: api-master
description: Automates API development workflows including endpoint generation, schema validation, documentation, and testing for REST, GraphQL, and RPC APIs. Claude activates this when working with API code.
---

# API-MASTER - API Development Acceleration Skill

Automatically assists with API development, from scaffolding to documentation to testing.

## What API-MASTER does for Claude

**Intelligent Detection**: Automatically recognizes API code and patterns
**Code Generation**: Creates endpoints, models, and tests with best practices
**Schema Management**: Validates schemas and detects breaking changes
**Documentation**: Auto-generates OpenAPI/Swagger specs and API docs
**Best Practices**: Enforces REST/GraphQL standards and security patterns
**Multi-Framework**: Works with FastAPI, Express, Django REST, Flask, and more

## When Claude activates API-MASTER

Claude automatically uses this skill when:

**File Patterns Detected:**
- Working in `api/`, `routes/`, `controllers/`, `graphql/` directories
- Editing files like `*_controller.py`, `*_routes.js`, `*_handler.go`
- Modifying API schema or OpenAPI specification files

**Code Patterns Detected:**
- REST decorators: `@app.route`, `@router.get`, `app.post()`
- API classes: `APIView`, `ViewSet`, `RestController`
- GraphQL patterns: `type Query`, `type Mutation`, resolvers
- API frameworks: FastAPI, Express, Django REST, Flask-RESTFUL

**User Keywords:**
- "API", "endpoint", "REST", "GraphQL"
- "OpenAPI", "Swagger", "schema"
- "request validation", "API docs"
- "API client", "API testing"

## How API-MASTER works

### 1. Detection & Analysis
- Identifies API framework and patterns
- Analyzes existing endpoint structure
- Detects schema definitions
- Maps routes and handlers

### 2. Code Generation
- Creates CRUD endpoints with best practices
- Generates request/response models with validation
- Adds pagination, filtering, sorting
- Implements proper error handling
- Creates API tests

### 3. Schema Management
- Validates request/response schemas
- Checks for breaking changes
- Generates TypeScript/Python types from schemas
- Converts between schema formats

### 4. Documentation
- Generates OpenAPI specifications
- Creates API reference documentation
- Builds interactive API explorers
- Maintains API changelog

### 5. Quality Assurance
- Enforces REST naming conventions
- Adds security headers and CORS
- Implements rate limiting
- Suggests caching strategies
- Validates API design patterns

## Configuration (optional)

Claude can use custom settings via `.api_master_config.json`:

```json
{
  "api_type": "rest",
  "framework": "auto",
  "auth_method": "jwt",
  "enable_auto_docs": true,
  "enable_auto_validation": true,
  "enable_auto_tests": true,
  "pagination_default": "offset",
  "error_format": "json_api",
  "versioning_strategy": "url",
  "rate_limiting": true,
  "enable_cors": true
}
```

## Supported Frameworks

**Python:**
- FastAPI (Pydantic models, async, auto-docs)
- Django REST Framework (ViewSets, serializers)
- Flask-RESTFUL (blueprints, decorators)

**JavaScript/TypeScript:**
- Express.js (middleware, routing)
- NestJS (decorators, modules)
- Fastify (schemas, plugins)

**Go:**
- Gin, Echo, Chi (handlers, middleware)

**GraphQL:**
- Apollo Server, GraphQL Yoga, Strawberry

API-MASTER accelerates API development while maintaining quality and consistency.
