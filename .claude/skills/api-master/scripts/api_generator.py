#!/usr/bin/env python3
"""
API Generator - Generate API boilerplate for REST and GraphQL APIs

Supports multiple frameworks:
- FastAPI (Python)
- Flask (Python)
- Django REST Framework (Python)
- Express.js (JavaScript/TypeScript)
- NestJS (TypeScript)

Usage:
    python api_generator.py --framework fastapi --resource User --output ./api/
    python api_generator.py --framework express --resource Product --crud
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class APIConfig:
    """API generation configuration"""
    framework: str
    resource: str
    output_dir: str
    crud: bool = True
    auth: bool = False
    validation: bool = True
    tests: bool = True
    docs: bool = True


class APIGenerator:
    """Base API generator"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.output_path = Path(config.output_dir)
        self.resource_lower = config.resource.lower()
        self.resource_plural = f"{self.resource_lower}s"  # Simple pluralization

    def generate(self):
        """Generate API code"""
        raise NotImplementedError

    def save_file(self, filename: str, content: str):
        """Save generated file"""
        filepath = self.output_path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            f.write(content)

        print(f"✓ Generated: {filepath}")


class FastAPIGenerator(APIGenerator):
    """FastAPI code generator"""

    def generate(self):
        """Generate FastAPI code"""
        print(f"\nGenerating FastAPI code for {self.config.resource}...")

        # Generate model
        model_code = self._generate_model()
        self.save_file(f"models/{self.resource_lower}.py", model_code)

        # Generate schema
        schema_code = self._generate_schema()
        self.save_file(f"schemas/{self.resource_lower}.py", schema_code)

        # Generate router
        router_code = self._generate_router()
        self.save_file(f"routers/{self.resource_lower}.py", router_code)

        # Generate CRUD operations
        crud_code = self._generate_crud()
        self.save_file(f"crud/{self.resource_lower}.py", crud_code)

        if self.config.tests:
            test_code = self._generate_tests()
            self.save_file(f"tests/test_{self.resource_lower}.py", test_code)

        print(f"\n✅ FastAPI code generated successfully!")
        print(f"\nNext steps:")
        print(f"1. Add router to main.py: app.include_router({self.resource_lower}.router)")
        print(f"2. Run: uvicorn main:app --reload")
        print(f"3. Visit: http://localhost:8000/docs")

    def _generate_model(self) -> str:
        return f'''"""
{self.config.resource} database model
"""
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from database import Base


class {self.config.resource}(Base):
    """
    {self.config.resource} model

    Represents a {self.resource_lower} in the system.
    """
    __tablename__ = "{self.resource_plural}"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<{self.config.resource}(id={{self.id}}, name='{{self.name}}')>"
'''

    def _generate_schema(self) -> str:
        return f'''"""
{self.config.resource} Pydantic schemas for request/response validation
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class {self.config.resource}Base(BaseModel):
    """Base {self.resource_lower} schema with common fields"""
    name: str = Field(..., min_length=1, max_length=100, description="{self.config.resource} name")
    description: Optional[str] = Field(None, max_length=500, description="{self.config.resource} description")


class {self.config.resource}Create(BaseModel):
    """Schema for creating a {self.resource_lower}"""
    model_config = ConfigDict(json_schema_extra={{
        "example": {{
            "name": "Example {self.config.resource}",
            "description": "This is an example {self.resource_lower}"
        }}
    }})


class {self.config.resource}Update(BaseModel):
    """Schema for updating a {self.resource_lower}"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class {self.config.resource}Response({self.config.resource}Base):
    """Schema for {self.resource_lower} responses"""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class {self.config.resource}List(BaseModel):
    """Schema for paginated {self.resource_lower} list"""
    items: list[{self.config.resource}Response]
    total: int
    page: int
    page_size: int
    pages: int
'''

    def _generate_router(self) -> str:
        return f'''"""
{self.config.resource} API router
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from database import get_db
from schemas.{self.resource_lower} import (
    {self.config.resource}Create,
    {self.config.resource}Update,
    {self.config.resource}Response,
    {self.config.resource}List
)
from crud.{self.resource_lower} import {self.resource_lower}_crud


router = APIRouter(
    prefix="/{self.resource_plural}",
    tags=["{self.resource_plural}"],
)


@router.post(
    "/",
    response_model={self.config.resource}Response,
    status_code=status.HTTP_201_CREATED,
    summary="Create a {self.resource_lower}",
    description="Create a new {self.resource_lower} with the provided data."
)
async def create_{self.resource_lower}(
    {self.resource_lower}: {self.config.resource}Create,
    db: Session = Depends(get_db)
):
    """Create a new {self.resource_lower}"""
    return {self.resource_lower}_crud.create(db=db, obj_in={self.resource_lower})


@router.get(
    "/",
    response_model={self.config.resource}List,
    summary="List {self.resource_plural}",
    description="Retrieve a paginated list of {self.resource_plural}."
)
async def list_{self.resource_plural}(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of items to return"),
    db: Session = Depends(get_db)
):
    """List {self.resource_plural} with pagination"""
    items = {self.resource_lower}_crud.get_multi(db=db, skip=skip, limit=limit)
    total = {self.resource_lower}_crud.count(db=db)

    return {{
        "items": items,
        "total": total,
        "page": skip // limit + 1,
        "page_size": limit,
        "pages": (total + limit - 1) // limit
    }}


@router.get(
    "/{{id}}",
    response_model={self.config.resource}Response,
    summary="Get {self.resource_lower} by ID",
    description="Retrieve a specific {self.resource_lower} by its ID."
)
async def get_{self.resource_lower}(
    id: int,
    db: Session = Depends(get_db)
):
    """Get {self.resource_lower} by ID"""
    {self.resource_lower} = {self.resource_lower}_crud.get(db=db, id=id)
    if not {self.resource_lower}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{self.config.resource} with id {{id}} not found"
        )
    return {self.resource_lower}


@router.put(
    "/{{id}}",
    response_model={self.config.resource}Response,
    summary="Update {self.resource_lower}",
    description="Update an existing {self.resource_lower}."
)
async def update_{self.resource_lower}(
    id: int,
    {self.resource_lower}_update: {self.config.resource}Update,
    db: Session = Depends(get_db)
):
    """Update {self.resource_lower}"""
    {self.resource_lower} = {self.resource_lower}_crud.get(db=db, id=id)
    if not {self.resource_lower}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{self.config.resource} with id {{id}} not found"
        )
    return {self.resource_lower}_crud.update(db=db, db_obj={self.resource_lower}, obj_in={self.resource_lower}_update)


@router.delete(
    "/{{id}}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete {self.resource_lower}",
    description="Delete a {self.resource_lower} by ID."
)
async def delete_{self.resource_lower}(
    id: int,
    db: Session = Depends(get_db)
):
    """Delete {self.resource_lower}"""
    {self.resource_lower} = {self.resource_lower}_crud.get(db=db, id=id)
    if not {self.resource_lower}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{self.config.resource} with id {{id}} not found"
        )
    {self.resource_lower}_crud.remove(db=db, id=id)
'''

    def _generate_crud(self) -> str:
        return f'''"""
CRUD operations for {self.config.resource}
"""
from typing import List, Optional
from sqlalchemy.orm import Session

from models.{self.resource_lower} import {self.config.resource}
from schemas.{self.resource_lower} import {self.config.resource}Create, {self.config.resource}Update


class {self.config.resource}CRUD:
    """CRUD operations for {self.config.resource}"""

    def get(self, db: Session, id: int) -> Optional[{self.config.resource}]:
        """Get {self.resource_lower} by ID"""
        return db.query({self.config.resource}).filter({self.config.resource}.id == id).first()

    def get_multi(
        self, db: Session, *, skip: int = 0, limit: int = 100
    ) -> List[{self.config.resource}]:
        """Get multiple {self.resource_plural} with pagination"""
        return db.query({self.config.resource}).offset(skip).limit(limit).all()

    def count(self, db: Session) -> int:
        """Count total {self.resource_plural}"""
        return db.query({self.config.resource}).count()

    def create(self, db: Session, *, obj_in: {self.config.resource}Create) -> {self.config.resource}:
        """Create a new {self.resource_lower}"""
        db_obj = {self.config.resource}(**obj_in.model_dump())
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self, db: Session, *, db_obj: {self.config.resource}, obj_in: {self.config.resource}Update
    ) -> {self.config.resource}:
        """Update {self.resource_lower}"""
        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def remove(self, db: Session, *, id: int) -> {self.config.resource}:
        """Delete {self.resource_lower}"""
        obj = db.query({self.config.resource}).get(id)
        db.delete(obj)
        db.commit()
        return obj


{self.resource_lower}_crud = {self.config.resource}CRUD()
'''

    def _generate_tests(self) -> str:
        return f'''"""
Tests for {self.config.resource} API
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


def test_create_{self.resource_lower}(client: TestClient, db: Session):
    """Test creating a {self.resource_lower}"""
    response = client.post(
        "/{self.resource_plural}/",
        json={{
            "name": "Test {self.config.resource}",
            "description": "Test description"
        }}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test {self.config.resource}"
    assert data["description"] == "Test description"
    assert "id" in data
    assert "created_at" in data


def test_list_{self.resource_plural}(client: TestClient, db: Session):
    """Test listing {self.resource_plural}"""
    response = client.get("/{self.resource_plural}/")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data


def test_get_{self.resource_lower}(client: TestClient, db: Session):
    """Test getting a {self.resource_lower} by ID"""
    # Create a {self.resource_lower} first
    create_response = client.post(
        "/{self.resource_plural}/",
        json={{"name": "Test {self.config.resource}", "description": "Test"}}
    )
    created_id = create_response.json()["id"]

    # Get the {self.resource_lower}
    response = client.get(f"/{self.resource_plural}/{{created_id}}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == created_id
    assert data["name"] == "Test {self.config.resource}"


def test_update_{self.resource_lower}(client: TestClient, db: Session):
    """Test updating a {self.resource_lower}"""
    # Create a {self.resource_lower} first
    create_response = client.post(
        "/{self.resource_plural}/",
        json={{"name": "Original Name", "description": "Original"}}
    )
    created_id = create_response.json()["id"]

    # Update the {self.resource_lower}
    response = client.put(
        f"/{self.resource_plural}/{{created_id}}",
        json={{"name": "Updated Name"}}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Name"


def test_delete_{self.resource_lower}(client: TestClient, db: Session):
    """Test deleting a {self.resource_lower}"""
    # Create a {self.resource_lower} first
    create_response = client.post(
        "/{self.resource_plural}/",
        json={{"name": "To Delete", "description": "Delete me"}}
    )
    created_id = create_response.json()["id"]

    # Delete the {self.resource_lower}
    response = client.delete(f"/{self.resource_plural}/{{created_id}}")
    assert response.status_code == 204

    # Verify it's deleted
    get_response = client.get(f"/{self.resource_plural}/{{created_id}}")
    assert get_response.status_code == 404


def test_get_nonexistent_{self.resource_lower}(client: TestClient):
    """Test getting a non-existent {self.resource_lower}"""
    response = client.get("/{self.resource_plural}/99999")
    assert response.status_code == 404
    assert "{self.config.resource}" in response.json()["detail"]
'''


class ExpressGenerator(APIGenerator):
    """Express.js/TypeScript code generator"""

    def generate(self):
        """Generate Express code"""
        print(f"\nGenerating Express.js code for {self.config.resource}...")

        # Generate model
        model_code = self._generate_model()
        self.save_file(f"models/{self.resource_lower}.model.ts", model_code)

        # Generate controller
        controller_code = self._generate_controller()
        self.save_file(f"controllers/{self.resource_lower}.controller.ts", controller_code)

        # Generate routes
        routes_code = self._generate_routes()
        self.save_file(f"routes/{self.resource_lower}.routes.ts", routes_code)

        # Generate validation
        validation_code = self._generate_validation()
        self.save_file(f"validators/{self.resource_lower}.validator.ts", validation_code)

        if self.config.tests:
            test_code = self._generate_tests()
            self.save_file(f"tests/{self.resource_lower}.test.ts", test_code)

        print(f"\n✅ Express.js code generated successfully!")
        print(f"\nNext steps:")
        print(f"1. Import router in app.ts: import {self.resource_lower}Routes from './routes/{self.resource_lower}.routes'")
        print(f"2. Add router: app.use('/api/{self.resource_plural}', {self.resource_lower}Routes)")
        print(f"3. Run: npm run dev")

    def _generate_model(self) -> str:
        return f'''/**
 * {self.config.resource} model
 */

export interface I{self.config.resource} {{
  id: number;
  name: string;
  description?: string;
  createdAt: Date;
  updatedAt?: Date;
}}

export interface Create{self.config.resource}DTO {{
  name: string;
  description?: string;
}}

export interface Update{self.config.resource}DTO {{
  name?: string;
  description?: string;
}}

export class {self.config.resource} implements I{self.config.resource} {{
  constructor(
    public id: number,
    public name: string,
    public createdAt: Date,
    public description?: string,
    public updatedAt?: Date
  ) {{}}
}}
'''

    def _generate_controller(self) -> str:
        return f'''/**
 * {self.config.resource} controller
 */
import {{ Request, Response, NextFunction }} from 'express';
import {{ Create{self.config.resource}DTO, Update{self.config.resource}DTO }} from '../models/{self.resource_lower}.model';

// Mock database - replace with actual database
const {self.resource_plural}: any[] = [];
let nextId = 1;

export class {self.config.resource}Controller {{
  /**
   * Create a new {self.resource_lower}
   */
  static async create(req: Request, res: Response, next: NextFunction) {{
    try {{
      const data: Create{self.config.resource}DTO = req.body;

      const new{self.config.resource} = {{
        id: nextId++,
        ...data,
        createdAt: new Date()
      }};

      {self.resource_plural}.push(new{self.config.resource});

      res.status(201).json(new{self.config.resource});
    }} catch (error) {{
      next(error);
    }}
  }}

  /**
   * List all {self.resource_plural} with pagination
   */
  static async list(req: Request, res: Response, next: NextFunction) {{
    try {{
      const page = parseInt(req.query.page as string) || 1;
      const limit = parseInt(req.query.limit as string) || 20;
      const skip = (page - 1) * limit;

      const items = {self.resource_plural}.slice(skip, skip + limit);
      const total = {self.resource_plural}.length;

      res.json({{
        items,
        total,
        page,
        pageSize: limit,
        pages: Math.ceil(total / limit)
      }});
    }} catch (error) {{
      next(error);
    }}
  }}

  /**
   * Get {self.resource_lower} by ID
   */
  static async getById(req: Request, res: Response, next: NextFunction) {{
    try {{
      const id = parseInt(req.params.id);
      const {self.resource_lower} = {self.resource_plural}.find(u => u.id === id);

      if (!{self.resource_lower}) {{
        return res.status(404).json({{
          error: '{self.config.resource} not found',
          message: `{self.config.resource} with id ${{id}} not found`
        }});
      }}

      res.json({self.resource_lower});
    }} catch (error) {{
      next(error);
    }}
  }}

  /**
   * Update {self.resource_lower}
   */
  static async update(req: Request, res: Response, next: NextFunction) {{
    try {{
      const id = parseInt(req.params.id);
      const updates: Update{self.config.resource}DTO = req.body;

      const index = {self.resource_plural}.findIndex(u => u.id === id);
      if (index === -1) {{
        return res.status(404).json({{
          error: '{self.config.resource} not found',
          message: `{self.config.resource} with id ${{id}} not found`
        }});
      }}

      {self.resource_plural}[index] = {{
        ...{self.resource_plural}[index],
        ...updates,
        updatedAt: new Date()
      }};

      res.json({self.resource_plural}[index]);
    }} catch (error) {{
      next(error);
    }}
  }}

  /**
   * Delete {self.resource_lower}
   */
  static async delete(req: Request, res: Response, next: NextFunction) {{
    try {{
      const id = parseInt(req.params.id);
      const index = {self.resource_plural}.findIndex(u => u.id === id);

      if (index === -1) {{
        return res.status(404).json({{
          error: '{self.config.resource} not found',
          message: `{self.config.resource} with id ${{id}} not found`
        }});
      }}

      {self.resource_plural}.splice(index, 1);
      res.status(204).send();
    }} catch (error) {{
      next(error);
    }}
  }}
}}
'''

    def _generate_routes(self) -> str:
        return f'''/**
 * {self.config.resource} routes
 */
import {{ Router }} from 'express';
import {{ {self.config.resource}Controller }} from '../controllers/{self.resource_lower}.controller';
import {{ validate{self.config.resource}Create, validate{self.config.resource}Update }} from '../validators/{self.resource_lower}.validator';

const router = Router();

// POST /{self.resource_plural} - Create a {self.resource_lower}
router.post('/', validate{self.config.resource}Create, {self.config.resource}Controller.create);

// GET /{self.resource_plural} - List {self.resource_plural}
router.get('/', {self.config.resource}Controller.list);

// GET /{self.resource_plural}/:id - Get {self.resource_lower} by ID
router.get('/:id', {self.config.resource}Controller.getById);

// PUT /{self.resource_plural}/:id - Update {self.resource_lower}
router.put('/:id', validate{self.config.resource}Update, {self.config.resource}Controller.update);

// DELETE /{self.resource_plural}/:id - Delete {self.resource_lower}
router.delete('/:id', {self.config.resource}Controller.delete);

export default router;
'''

    def _generate_validation(self) -> str:
        return f'''/**
 * {self.config.resource} validation middleware
 */
import {{ Request, Response, NextFunction }} from 'express';
import Joi from 'joi';

const create{self.config.resource}Schema = Joi.object({{
  name: Joi.string().min(1).max(100).required(),
  description: Joi.string().max(500).optional()
}});

const update{self.config.resource}Schema = Joi.object({{
  name: Joi.string().min(1).max(100).optional(),
  description: Joi.string().max(500).optional()
}});

export function validate{self.config.resource}Create(req: Request, res: Response, next: NextFunction) {{
  const {{ error }} = create{self.config.resource}Schema.validate(req.body);

  if (error) {{
    return res.status(400).json({{
      error: 'Validation error',
      details: error.details.map(d => d.message)
    }});
  }}

  next();
}}

export function validate{self.config.resource}Update(req: Request, res: Response, next: NextFunction) {{
  const {{ error }} = update{self.config.resource}Schema.validate(req.body);

  if (error) {{
    return res.status(400).json({{
      error: 'Validation error',
      details: error.details.map(d => d.message)
    }});
  }}

  next();
}}
'''

    def _generate_tests(self) -> str:
        return f'''/**
 * {self.config.resource} API tests
 */
import request from 'supertest';
import app from '../app';

describe('{self.config.resource} API', () => {{
  let created{self.config.resource}Id: number;

  describe('POST /{self.resource_plural}', () => {{
    it('should create a new {self.resource_lower}', async () => {{
      const response = await request(app)
        .post('/{self.resource_plural}')
        .send({{
          name: 'Test {self.config.resource}',
          description: 'Test description'
        }});

      expect(response.status).toBe(201);
      expect(response.body.name).toBe('Test {self.config.resource}');
      expect(response.body.id).toBeDefined();

      created{self.config.resource}Id = response.body.id;
    }});

    it('should return 400 for invalid data', async () => {{
      const response = await request(app)
        .post('/{self.resource_plural}')
        .send({{ name: '' }});

      expect(response.status).toBe(400);
    }});
  }});

  describe('GET /{self.resource_plural}', () => {{
    it('should list {self.resource_plural}', async () => {{
      const response = await request(app).get('/{self.resource_plural}');

      expect(response.status).toBe(200);
      expect(response.body.items).toBeDefined();
      expect(response.body.total).toBeGreaterThan(0);
    }});
  }});

  describe('GET /{self.resource_plural}/:id', () => {{
    it('should get {self.resource_lower} by id', async () => {{
      const response = await request(app)
        .get(`/{self.resource_plural}/${{created{self.config.resource}Id}}`);

      expect(response.status).toBe(200);
      expect(response.body.id).toBe(created{self.config.resource}Id);
    }});

    it('should return 404 for non-existent id', async () => {{
      const response = await request(app).get('/{self.resource_plural}/99999');

      expect(response.status).toBe(404);
    }});
  }});

  describe('PUT /{self.resource_plural}/:id', () => {{
    it('should update {self.resource_lower}', async () => {{
      const response = await request(app)
        .put(`/{self.resource_plural}/${{created{self.config.resource}Id}}`)
        .send({{ name: 'Updated Name' }});

      expect(response.status).toBe(200);
      expect(response.body.name).toBe('Updated Name');
    }});
  }});

  describe('DELETE /{self.resource_plural}/:id', () => {{
    it('should delete {self.resource_lower}', async () => {{
      const response = await request(app)
        .delete(`/{self.resource_plural}/${{created{self.config.resource}Id}}`);

      expect(response.status).toBe(204);
    }});
  }});
}});
'''


def main():
    parser = argparse.ArgumentParser(
        description="Generate API boilerplate code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate FastAPI code
  python api_generator.py --framework fastapi --resource User --output ./api/

  # Generate Express.js code
  python api_generator.py --framework express --resource Product --output ./src/

  # Generate without tests
  python api_generator.py --framework fastapi --resource Post --no-tests
        """
    )

    parser.add_argument(
        '--framework',
        choices=['fastapi', 'express', 'flask', 'django'],
        required=True,
        help='API framework to use'
    )

    parser.add_argument(
        '--resource',
        required=True,
        help='Resource name (e.g., User, Product, Post)'
    )

    parser.add_argument(
        '--output',
        default='./generated',
        help='Output directory (default: ./generated)'
    )

    parser.add_argument(
        '--no-crud',
        action='store_true',
        help='Skip CRUD operations generation'
    )

    parser.add_argument(
        '--no-tests',
        action='store_true',
        help='Skip test generation'
    )

    parser.add_argument(
        '--no-auth',
        action='store_true',
        help='Skip authentication'
    )

    args = parser.parse_args()

    # Create config
    config = APIConfig(
        framework=args.framework,
        resource=args.resource,
        output_dir=args.output,
        crud=not args.no_crud,
        tests=not args.no_tests,
        auth=not args.no_auth
    )

    # Select generator
    generators = {
        'fastapi': FastAPIGenerator,
        'express': ExpressGenerator,
    }

    generator_class = generators.get(args.framework)
    if not generator_class:
        print(f"❌ Framework '{args.framework}' not yet implemented")
        print(f"Available: {', '.join(generators.keys())}")
        return 1

    # Generate code
    generator = generator_class(config)
    generator.generate()

    return 0


if __name__ == '__main__':
    exit(main())
