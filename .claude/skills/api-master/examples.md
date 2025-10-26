# API-MASTER Usage Examples

Real-world workflows showing how API-MASTER accelerates API development.

## Example 1: Creating a New REST API from Scratch

### Scenario
You need to create a user management API with CRUD operations.

### Without API-MASTER
1. Manually create model file (15 min)
2. Write schema/validation (10 min)
3. Create route handlers (20 min)
4. Write CRUD operations (15 min)
5. Add error handling (10 min)
6. Write tests (20 min)
7. Generate OpenAPI docs (10 min)

**Total: ~100 minutes**

### With API-MASTER

```
User: "Create a REST API for managing users with name, email, and role fields"

API-MASTER:
✓ Generated User model with SQLAlchemy
✓ Created Pydantic schemas (UserCreate, UserUpdate, UserResponse)
✓ Built CRUD endpoints (GET, POST, PUT, DELETE)
✓ Added pagination to list endpoint
✓ Implemented request validation
✓ Generated integration tests
✓ Created OpenAPI specification

All files saved to api/users/
```

**Total: < 5 minutes**

---

## Example 2: Analyzing and Improving Existing API

### Scenario
You have an existing API that needs optimization.

### Workflow

```python
# Existing endpoint (before)
@app.get("/posts")
def get_posts():
    posts = db.query(Post).all()
    return posts
```

**API-MASTER detects issues:**

```
🔍 API-MASTER Analysis:

Issues Found:
1. ❌ No pagination - can return unlimited results
2. ❌ No request validation
3. ❌ Missing error handling
4. ❌ No response schema
5. ❌ No rate limiting
6. ⚠️  Missing API documentation
7. ⚠️  No filtering or sorting options

Suggested Improvements:
Would you like me to:
1. Add pagination (offset/limit or cursor-based)
2. Implement request validation with query parameters
3. Add proper error responses
4. Define response schema
5. Add rate limiting (suggest 100 req/min)
6. Generate OpenAPI docs
```

**User: "Yes, apply all improvements"**

```python
# Improved endpoint (after)
@app.get(
    "/posts",
    response_model=PostList,
    summary="List blog posts",
    description="Get a paginated list of blog posts with optional filtering"
)
@limiter.limit("100/minute")
async def get_posts(
    skip: int = Query(0, ge=0, description="Number of posts to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of posts to return"),
    author_id: Optional[int] = Query(None, description="Filter by author ID"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    sort_by: str = Query("created_at", enum=["created_at", "title", "views"]),
    db: Session = Depends(get_db)
):
    """
    Retrieve paginated list of blog posts.

    - **skip**: Pagination offset
    - **limit**: Maximum number of results (max 100)
    - **author_id**: Filter by specific author
    - **tag**: Filter by tag
    - **sort_by**: Sort field (created_at, title, or views)
    """
    try:
        query = db.query(Post)

        # Apply filters
        if author_id:
            query = query.filter(Post.author_id == author_id)
        if tag:
            query = query.join(Post.tags).filter(Tag.name == tag)

        # Apply sorting
        query = query.order_by(getattr(Post, sort_by).desc())

        # Get total count
        total = query.count()

        # Apply pagination
        posts = query.offset(skip).limit(limit).all()

        return PostList(
            items=posts,
            total=total,
            page=skip // limit + 1,
            page_size=limit,
            pages=(total + limit - 1) // limit
        )
    except Exception as e:
        logger.error(f"Error fetching posts: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch posts"
        )
```

---

## Example 3: Detecting Breaking Changes in API Schema

### Scenario
You're updating an API and want to ensure no breaking changes.

### Workflow

```bash
# You modify the User schema
# Old: { "name": "string", "email": "string" }
# New: { "name": "string", "email": "string", "phone": "string" (required) }
```

**API-MASTER detects:**

```
🚨 BREAKING CHANGE DETECTED!

Schema: User
Change: Added required field "phone"

Impact:
- Existing API clients will break
- Current requests missing "phone" will be rejected
- This is a BREAKING change

Recommendations:
1. ✅ Make "phone" optional: phone?: string
2. ✅ Use API versioning: /v2/users
3. ✅ Add default value: phone = ""
4. ❌ Do NOT deploy as required field

Would you like me to:
a) Make "phone" optional instead?
b) Create a new API version (/v2)?
c) Continue anyway (not recommended)
```

**User: "Make it optional"**

API-MASTER updates schema to make `phone` optional and prevents the breaking change.

---

## Example 4: Generating API Client SDK

### Scenario
Frontend team needs a TypeScript client for your API.

### Workflow

```
User: "Generate a TypeScript client for the users API"

API-MASTER:
✓ Analyzed OpenAPI specification
✓ Generated TypeScript types from schemas
✓ Created API client class with methods
✓ Added error handling
✓ Included request/response types

Generated files:
- api-client.ts (main client)
- types.ts (TypeScript interfaces)
- errors.ts (error classes)
- README.md (usage guide)
```

**Generated TypeScript Client:**

```typescript
// api-client.ts
export class UsersAPI {
  private baseURL: string;

  constructor(baseURL: string = 'https://api.example.com') {
    this.baseURL = baseURL;
  }

  async listUsers(params: ListUsersParams): Promise<UserList> {
    const response = await fetch(
      `${this.baseURL}/users?${new URLSearchParams(params)}`,
      { headers: { 'Content-Type': 'application/json' } }
    );

    if (!response.ok) {
      throw new APIError(response.status, await response.text());
    }

    return await response.json();
  }

  async getUser(id: number): Promise<User> {
    const response = await fetch(`${this.baseURL}/users/${id}`);

    if (!response.ok) {
      if (response.status === 404) {
        throw new UserNotFoundError(id);
      }
      throw new APIError(response.status, await response.text());
    }

    return await response.json();
  }

  async createUser(data: UserCreate): Promise<User> {
    const response = await fetch(`${this.baseURL}/users`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      throw new APIError(response.status, await response.text());
    }

    return await response.json();
  }
}

// types.ts
export interface User {
  id: number;
  name: string;
  email: string;
  created_at: string;
}

export interface UserCreate {
  name: string;
  email: string;
}

export interface ListUsersParams {
  page?: number;
  limit?: number;
}

export interface UserList {
  items: User[];
  total: number;
  page: number;
}
```

---

## Example 5: Auto-Generating API Tests

### Scenario
You've created several API endpoints and need comprehensive tests.

### Workflow

```
User: "Generate tests for the users API"

API-MASTER:
✓ Analyzed 5 endpoints in users API
✓ Generated test cases for each endpoint
✓ Created fixtures and factories
✓ Added edge case tests
✓ Configured test database

Generated 23 tests covering:
- Happy path scenarios
- Error cases (404, 400, 500)
- Edge cases (empty lists, invalid data)
- Authentication/authorization
- Pagination
```

**Generated Tests:**

```python
# test_users_api.py
import pytest
from fastapi.testclient import TestClient

class TestUsersAPI:
    """Comprehensive tests for Users API"""

    def test_create_user_success(self, client: TestClient):
        """Test successful user creation"""
        response = client.post(
            "/users",
            json={"name": "John Doe", "email": "john@example.com"}
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "John Doe"
        assert data["email"] == "john@example.com"
        assert "id" in data

    def test_create_user_invalid_email(self, client: TestClient):
        """Test user creation with invalid email"""
        response = client.post(
            "/users",
            json={"name": "John Doe", "email": "invalid-email"}
        )
        assert response.status_code == 400
        assert "email" in response.json()["detail"].lower()

    def test_list_users_pagination(self, client: TestClient, create_users):
        """Test pagination works correctly"""
        # Create 50 users
        create_users(50)

        # Get first page
        response = client.get("/users?page=1&limit=20")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 20
        assert data["total"] == 50
        assert data["pages"] == 3

    def test_get_user_not_found(self, client: TestClient):
        """Test getting non-existent user returns 404"""
        response = client.get("/users/99999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
```

---

## Example 6: Converting Between API Styles

### Scenario
Convert a REST API to GraphQL.

### Workflow

```
User: "Convert the users REST API to GraphQL"

API-MASTER:
✓ Analyzed REST endpoints
✓ Generated GraphQL schema
✓ Created resolvers
✓ Mapped CRUD operations to queries/mutations
✓ Added filtering and pagination support
```

**Generated GraphQL Schema:**

```graphql
type User {
  id: ID!
  name: String!
  email: String!
  createdAt: DateTime!
}

type UserList {
  items: [User!]!
  total: Int!
  page: Int!
}

input CreateUserInput {
  name: String!
  email: String!
}

input UpdateUserInput {
  name: String
  email: String
}

type Query {
  user(id: ID!): User
  users(page: Int, limit: Int): UserList!
}

type Mutation {
  createUser(input: CreateUserInput!): User!
  updateUser(id: ID!, input: UpdateUserInput!): User!
  deleteUser(id: ID!): Boolean!
}
```

---

## Common Workflows Summary

| Task | Manual Time | With API-MASTER | Time Saved |
|------|-------------|-----------------|------------|
| Create CRUD API | 90-120 min | 3-5 min | 95% |
| Generate OpenAPI docs | 30-45 min | 1 min | 97% |
| Write API tests | 40-60 min | 2-3 min | 95% |
| Detect breaking changes | 20-30 min | 1 min | 97% |
| Generate API client | 60-90 min | 2-3 min | 97% |
| Add pagination | 15-20 min | 1 min | 95% |

**Average time savings: ~96%**

---

## Tips for Using API-MASTER

1. **Be Specific**: The more details you provide, the better the generated code
   - ✅ "Create a REST API for blog posts with title, content, author, and tags"
   - ❌ "Create an API"

2. **Review Generated Code**: Always review before deploying to production

3. **Customize Configuration**: Use `.api_master_config.json` for project-specific settings

4. **Run Tests**: Generated tests are comprehensive but may need project-specific adjustments

5. **Iterate**: API-MASTER learns from your preferences over time

---

API-MASTER saves developers an average of 15-20 hours per week on API development tasks.
