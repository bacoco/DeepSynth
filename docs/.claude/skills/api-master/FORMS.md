# API-MASTER Configuration Forms

## Configuration File: `.api_master_config.json`

Place this file in your project root to customize API-MASTER behavior.

## Configuration Options

### API Type
```json
{
  "api_type": "rest" | "graphql" | "grpc" | "mixed"
}
```
- **rest**: REST/RESTful APIs
- **graphql**: GraphQL APIs
- **grpc**: gRPC services
- **mixed**: Multiple API styles in same project

### Framework Detection
```json
{
  "framework": "auto" | "fastapi" | "express" | "django" | "flask" | "nestjs" | "spring"
}
```
- **auto**: Automatically detect framework (recommended)
- **specific**: Force a specific framework

### Authentication Method
```json
{
  "auth_method": "none" | "jwt" | "oauth2" | "api_key" | "basic" | "custom"
}
```
Determines what auth patterns to use in generated code.

### Documentation
```json
{
  "enable_auto_docs": true | false,
  "docs_format": "openapi" | "asyncapi" | "graphql_schema",
  "docs_output_path": "./docs/api"
}
```

### Validation
```json
{
  "enable_auto_validation": true | false,
  "validation_library": "auto" | "pydantic" | "joi" | "zod" | "marshmallow"
}
```

### Testing
```json
{
  "enable_auto_tests": true | false,
  "test_framework": "auto" | "pytest" | "jest" | "mocha" | "unittest",
  "test_coverage_target": 80
}
```

### Pagination
```json
{
  "pagination_default": "offset" | "cursor" | "page",
  "pagination_max_limit": 100,
  "pagination_default_limit": 20
}
```

### Error Handling
```json
{
  "error_format": "json_api" | "problem_json" | "custom",
  "include_stack_traces": false,
  "error_code_prefix": "API_"
}
```

### Versioning
```json
{
  "versioning_strategy": "url" | "header" | "none",
  "current_version": "v1",
  "version_prefix": "/api/v"
}
```

### Security
```json
{
  "rate_limiting": true | false,
  "rate_limit_default": "100/minute",
  "enable_cors": true | false,
  "cors_origins": ["*"],
  "security_headers": true
}
```

### Code Generation
```json
{
  "code_style": "auto" | "strict" | "minimal",
  "generate_docstrings": true,
  "generate_type_hints": true,
  "async_by_default": true
}
```

## Example Configurations

### FastAPI Project
```json
{
  "api_type": "rest",
  "framework": "fastapi",
  "auth_method": "jwt",
  "enable_auto_docs": true,
  "enable_auto_validation": true,
  "enable_auto_tests": true,
  "pagination_default": "offset",
  "error_format": "json_api",
  "versioning_strategy": "url",
  "rate_limiting": true,
  "enable_cors": true,
  "async_by_default": true
}
```

### Express.js Project
```json
{
  "api_type": "rest",
  "framework": "express",
  "auth_method": "jwt",
  "enable_auto_docs": true,
  "enable_auto_validation": true,
  "validation_library": "joi",
  "enable_auto_tests": true,
  "test_framework": "jest",
  "pagination_default": "cursor",
  "error_format": "json_api",
  "versioning_strategy": "url"
}
```

### GraphQL Project
```json
{
  "api_type": "graphql",
  "framework": "auto",
  "auth_method": "jwt",
  "enable_auto_docs": true,
  "docs_format": "graphql_schema",
  "enable_auto_tests": true,
  "error_format": "custom",
  "rate_limiting": true
}
```

### Minimal Configuration (Defaults)
```json
{
  "framework": "auto"
}
```

API-MASTER will use sensible defaults when configuration is not provided.
