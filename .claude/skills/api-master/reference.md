# API-MASTER Technical Reference

Complete technical reference for the API-MASTER skill.

## Activation Patterns

### File Patterns
API-MASTER activates when working with these file patterns:

```
**/api/**/*.{py,js,ts,go,java}
**/routes/**/*.{py,js,ts}
**/controllers/**/*.{py,js,ts}
**/graphql/**/*.{py,js,ts}
**/*_controller.{py,rb}
**/handlers/**/*.go
openapi.{yaml,yml,json}
swagger.{yaml,yml,json}
```

### Code Patterns
Detects API code through these patterns:

**Python (FastAPI)**
```python
@app.route(, @router.get(, @api.route(
@app.get(, @app.post(, @app.put(, @app.delete(
class APIRouter
from fastapi import
```

**JavaScript/TypeScript (Express)**
```javascript
app.get(, app.post(, router.use(
app.route(, express.Router(
```

**Go**
```go
http.HandleFunc(
mux.Handle(
gin.GET(, gin.POST(
```

**GraphQL**
```graphql
type Query, type Mutation
resolver, schema
```

### Keywords
Triggers on these keywords in user requests:
- API, endpoint, REST, GraphQL, RPC
- OpenAPI, Swagger, schema
- request, response, validation
- rate limiting, caching
- API client, SDK

## Core Capabilities

### 1. API Pattern Detection

**Framework Detection**
```python
def detect_framework(file_content: str) -> str:
    """Detect API framework from code"""
    patterns = {
        'fastapi': ['from fastapi import', '@app.get(', 'APIRouter'],
        'flask': ['from flask import', '@app.route(', 'Flask(__name__)'],
        'django': ['from django.', 'class.*APIView', 'ModelViewSet'],
        'express': ['express()', 'app.get(', 'router.use('],
    }
    # Detection logic...
```

**Endpoint Discovery**
- Scans files for route definitions
- Extracts HTTP methods and paths
- Identifies request/response models
- Maps authentication requirements

### 2. Code Generation

**Endpoint Template Variables**
```python
{
    'RESOURCE': 'User',           # Pascal case
    'resource': 'user',           # lowercase
    'resources': 'users',         # plural
    'RESOURCES': 'Users',         # Pascal plural
    'table_name': 'users',        # database table
}
```

**Generation Options**
```json
{
  "include_validation": true,
  "include_tests": true,
  "include_docs": true,
  "async_handlers": true,
  "add_rate_limiting": true,
  "pagination_style": "offset"
}
```

### 3. Schema Validation

**Validation Checks**
- ✅ Required fields present
- ✅ Type correctness
- ✅ Format validation (email, URL, date)
- ✅ Range constraints (min/max)
- ✅ Pattern matching (regex)
- ✅ Enum validation
- ✅ Nested object validation

**Breaking Change Detection**
```python
BREAKING_CHANGES = [
    'required_field_added',      # New required field
    'field_removed',             # Field deleted
    'type_changed',              # Type modified
    'required_changed',          # Optional → Required
    'endpoint_removed',          # Endpoint deleted
    'method_removed',            # HTTP method removed
]

NON_BREAKING_CHANGES = [
    'optional_field_added',      # New optional field
    'description_updated',       # Documentation change
    'default_added',             # Default value added
    'required_removed',          # Required → Optional
]
```

### 4. Documentation Generation

**OpenAPI Spec Structure**
```yaml
openapi: 3.0.3
info: # API metadata
servers: # Server URLs
tags: # Endpoint grouping
paths: # All endpoints
  /{resource}:
    get: # List operation
    post: # Create operation
  /{resource}/{id}:
    get: # Retrieve operation
    put: # Update operation
    delete: # Delete operation
components:
  schemas: # Data models
  securitySchemes: # Auth methods
```

### 5. Best Practices

**REST API Conventions**
```
GET    /users          → List users (paginated)
GET    /users/{id}     → Get specific user
POST   /users          → Create user
PUT    /users/{id}     → Update user (full)
PATCH  /users/{id}     → Update user (partial)
DELETE /users/{id}     → Delete user
```

**HTTP Status Codes**
```
200 OK              → Successful GET, PUT, PATCH
201 Created         → Successful POST
204 No Content      → Successful DELETE
400 Bad Request     → Invalid input
401 Unauthorized    → Missing/invalid authentication
403 Forbidden       → Insufficient permissions
404 Not Found       → Resource doesn't exist
422 Unprocessable   → Validation errors
429 Too Many Req    → Rate limit exceeded
500 Server Error    → Internal error
```

**Error Response Format**
```json
{
  "error": "validation_error",
  "message": "Invalid input data",
  "details": [
    {
      "field": "email",
      "error": "Invalid email format"
    }
  ],
  "timestamp": "2025-10-25T12:00:00Z",
  "path": "/users",
  "request_id": "abc-123"
}
```

## Framework-Specific Features

### FastAPI

**Auto-Generated Features**
```python
# Automatic request validation
class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr  # Auto-validates email format

# Automatic OpenAPI docs
@app.get("/users", response_model=List[User])
# Creates OpenAPI spec automatically

# Dependency injection
async def get_current_user(
    token: str = Depends(oauth2_scheme)
) -> User:
    # Verify token...
```

### Express.js

**Middleware Patterns**
```javascript
// Request validation middleware
const validateUser = (req, res, next) => {
  const schema = Joi.object({
    name: Joi.string().min(1).max(100).required(),
    email: Joi.string().email().required()
  });

  const { error } = schema.validate(req.body);
  if (error) {
    return res.status(400).json({ error: error.details });
  }
  next();
};

app.post('/users', validateUser, createUser);
```

### GraphQL

**Resolver Patterns**
```javascript
const resolvers = {
  Query: {
    user: async (_, { id }, context) => {
      return await context.dataSources.userAPI.getUser(id);
    },
    users: async (_, { page, limit }, context) => {
      return await context.dataSources.userAPI.getUsers(page, limit);
    }
  },
  Mutation: {
    createUser: async (_, { input }, context) => {
      return await context.dataSources.userAPI.createUser(input);
    }
  }
};
```

## Configuration

### .api_master_config.json

**Full Configuration Schema**
```json
{
  "api_type": "rest",
  "framework": "auto",
  "auth_method": "jwt",
  "enable_auto_docs": true,
  "enable_auto_validation": true,
  "enable_auto_tests": true,
  "pagination_default": "offset",
  "pagination_max_limit": 100,
  "pagination_default_limit": 20,
  "error_format": "json_api",
  "versioning_strategy": "url",
  "current_version": "v1",
  "rate_limiting": true,
  "rate_limit_default": "100/minute",
  "enable_cors": true,
  "cors_origins": ["*"],
  "security_headers": true,
  "code_style": "strict",
  "generate_docstrings": true,
  "generate_type_hints": true,
  "async_by_default": true
}
```

## Helper Scripts

### api_generator.py
Generates complete API code from resource specifications.

**Usage:**
```bash
python api_generator.py --framework fastapi --resource User --output ./api/
```

### schema_validator.py
Validates OpenAPI specifications and detects breaking changes.

**Usage:**
```bash
# Validate specification
python schema_validator.py --spec openapi.yaml --validate

# Detect breaking changes
python schema_validator.py --old v1.yaml --new v2.yaml --breaking-changes
```

### openapi_builder.py
Generates OpenAPI specifications from code.

**Usage:**
```bash
# Extract from FastAPI
python openapi_builder.py --framework fastapi --input ./api --output openapi.yaml

# Create example spec
python openapi_builder.py --example --output example.yaml
```

## Templates

### Available Templates
1. **rest_endpoint.py** - Complete CRUD endpoint (FastAPI)
2. **openapi_spec.yaml** - OpenAPI 3.0 specification
3. **graphql_resolver.py** - GraphQL resolver (coming soon)
4. **api_client.py** - API client SDK (coming soon)

### Using Templates
```bash
# Copy and customize template
cp .claude/skills/api-master/templates/rest_endpoint.py ./api/users.py

# Replace placeholders
sed -i 's/{RESOURCE}/User/g' ./api/users.py
sed -i 's/{resource}/user/g' ./api/users.py
sed -i 's/{resources}/users/g' ./api/users.py
```

## Performance Optimization

### Caching Strategies
```python
# Response caching
@cache(expire=300)  # 5 minutes
@app.get("/users/{id}")
async def get_user(id: int):
    # ...

# Database query caching
@cache_query(expire=60)
def get_popular_posts():
    # ...
```

### Rate Limiting
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.get("/users")
@limiter.limit("100/minute")
async def list_users():
    # ...
```

### Pagination Optimization
```python
# Cursor-based (better for large datasets)
@app.get("/users")
async def list_users(cursor: Optional[str] = None, limit: int = 20):
    # More efficient than offset-based for large tables
    # ...

# Offset-based (simpler, good for small datasets)
@app.get("/users")
async def list_users(skip: int = 0, limit: int = 20):
    # Works well for most use cases
    # ...
```

## Testing

### Test Coverage Goals
- Unit tests: 90%+ coverage
- Integration tests: All endpoints
- E2E tests: Critical user flows
- Performance tests: Load testing

### Test Categories
1. **Happy Path** - Successful operations
2. **Validation** - Invalid input handling
3. **Authentication** - Auth/authz scenarios
4. **Edge Cases** - Boundary conditions
5. **Error Handling** - Error scenarios
6. **Performance** - Load and stress tests

## Troubleshooting

### Common Issues

**Issue: OpenAPI docs not generating**
```python
# Ensure response_model is specified
@app.get("/users", response_model=List[User])  # ✅ Correct
@app.get("/users")  # ❌ Missing response_model
```

**Issue: Validation not working**
```python
# Use Pydantic models
class UserCreate(BaseModel):  # ✅ Correct
    name: str

def create_user(data: dict):  # ❌ Won't validate
```

**Issue: CORS errors**
```python
# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
```

## Resources

- [OpenAPI Specification](https://spec.openapi.org/oas/latest.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [REST API Best Practices](https://restfulapi.net/)
- [OWASP API Security](https://owasp.org/www-project-api-security/)
- [GraphQL Best Practices](https://graphql.org/learn/best-practices/)

---

API-MASTER version 1.0.0
