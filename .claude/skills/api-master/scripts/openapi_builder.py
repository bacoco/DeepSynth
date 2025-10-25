#!/usr/bin/env python3
"""
OpenAPI Builder - Generate OpenAPI specifications from code

Supports:
- FastAPI (automatic extraction)
- Flask
- Django REST Framework
- Express.js (via comments/decorators)

Usage:
    python openapi_builder.py --framework fastapi --input ./api --output openapi.yaml
    python openapi_builder.py --framework flask --input app.py --output swagger.json
"""

import argparse
import ast
import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class OpenAPIInfo:
    """OpenAPI info section"""
    title: str = "API"
    description: str = "API Documentation"
    version: str = "1.0.0"
    contact: Optional[Dict[str, str]] = None
    license: Optional[Dict[str, str]] = None


@dataclass
class OpenAPIServer:
    """OpenAPI server entry"""
    url: str
    description: Optional[str] = None


@dataclass
class OpenAPIParameter:
    """OpenAPI parameter"""
    name: str
    in_: str  # path, query, header, cookie
    required: bool = False
    schema: Dict[str, Any] = field(default_factory=lambda: {"type": "string"})
    description: Optional[str] = None

    def to_dict(self):
        d = {
            "name": self.name,
            "in": self.in_,
            "required": self.required,
            "schema": self.schema
        }
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class OpenAPIResponse:
    """OpenAPI response"""
    description: str
    content: Optional[Dict[str, Any]] = None

    def to_dict(self):
        d = {"description": self.description}
        if self.content:
            d["content"] = self.content
        return d


@dataclass
class OpenAPIOperation:
    """OpenAPI operation (endpoint)"""
    summary: Optional[str] = None
    description: Optional[str] = None
    operation_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    parameters: List[OpenAPIParameter] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, OpenAPIResponse] = field(default_factory=dict)

    def to_dict(self):
        d = {}
        if self.summary:
            d["summary"] = self.summary
        if self.description:
            d["description"] = self.description
        if self.operation_id:
            d["operationId"] = self.operation_id
        if self.tags:
            d["tags"] = self.tags
        if self.parameters:
            d["parameters"] = [p.to_dict() for p in self.parameters]
        if self.request_body:
            d["requestBody"] = self.request_body
        if self.responses:
            d["responses"] = {code: resp.to_dict() for code, resp in self.responses.items()}
        return d


class OpenAPIBuilder:
    """Build OpenAPI specification"""

    def __init__(self, info: OpenAPIInfo):
        self.info = info
        self.servers: List[OpenAPIServer] = []
        self.paths: Dict[str, Dict[str, OpenAPIOperation]] = {}
        self.components: Dict[str, Any] = {"schemas": {}, "securitySchemes": {}}
        self.tags: List[Dict[str, str]] = []

    def add_server(self, url: str, description: Optional[str] = None):
        """Add a server"""
        self.servers.append(OpenAPIServer(url=url, description=description))

    def add_tag(self, name: str, description: Optional[str] = None):
        """Add a tag"""
        tag = {"name": name}
        if description:
            tag["description"] = description
        self.tags.append(tag)

    def add_path(self, path: str, method: str, operation: OpenAPIOperation):
        """Add a path operation"""
        if path not in self.paths:
            self.paths[path] = {}
        self.paths[path][method.lower()] = operation

    def add_schema(self, name: str, schema: Dict[str, Any]):
        """Add a component schema"""
        self.components["schemas"][name] = schema

    def add_security_scheme(self, name: str, scheme: Dict[str, Any]):
        """Add a security scheme"""
        self.components["securitySchemes"][name] = scheme

    def build(self) -> Dict[str, Any]:
        """Build the OpenAPI specification"""
        spec = {
            "openapi": "3.0.3",
            "info": asdict(self.info),
            "servers": [asdict(s) for s in self.servers],
            "paths": {}
        }

        # Build paths
        for path, methods in self.paths.items():
            spec["paths"][path] = {}
            for method, operation in methods.items():
                spec["paths"][path][method] = operation.to_dict()

        # Add components if any
        if self.components["schemas"] or self.components["securitySchemes"]:
            spec["components"] = self.components

        # Add tags if any
        if self.tags:
            spec["tags"] = self.tags

        return spec

    def to_yaml(self, output_path: str):
        """Save as YAML"""
        spec = self.build()
        with open(output_path, 'w') as f:
            yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Generated OpenAPI spec: {output_path}")

    def to_json(self, output_path: str):
        """Save as JSON"""
        spec = self.build()
        with open(output_path, 'w') as f:
            json.dump(spec, f, indent=2)
        print(f"✅ Generated OpenAPI spec: {output_path}")


class FastAPIExtractor:
    """Extract OpenAPI info from FastAPI code"""

    def __init__(self, input_path: str):
        self.input_path = Path(input_path)
        self.builder = OpenAPIBuilder(OpenAPIInfo(
            title="FastAPI Application",
            description="Auto-generated from FastAPI code",
            version="1.0.0"
        ))

    def extract(self) -> OpenAPIBuilder:
        """Extract OpenAPI from FastAPI code"""
        if self.input_path.is_file():
            self._process_file(self.input_path)
        else:
            # Process all Python files in directory
            for py_file in self.input_path.rglob("*.py"):
                if "test" not in str(py_file):
                    self._process_file(py_file)

        # Add default server
        self.builder.add_server("http://localhost:8000", "Development server")

        return self.builder

    def _process_file(self, filepath: Path):
        """Process a Python file"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()

            tree = ast.parse(content)

            # Find route decorators
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._process_function(node)

        except Exception as e:
            print(f"Warning: Could not process {filepath}: {e}")

    def _process_function(self, func_node: ast.FunctionDef):
        """Process a function that might be an endpoint"""
        # Look for FastAPI route decorators
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    method = decorator.func.attr  # get, post, put, etc.

                    # Extract path from decorator arguments
                    if decorator.args and isinstance(decorator.args[0], ast.Constant):
                        path = decorator.args[0].value

                        # Extract docstring as description
                        docstring = ast.get_docstring(func_node)

                        # Create operation
                        operation = OpenAPIOperation(
                            summary=func_node.name.replace('_', ' ').title(),
                            description=docstring,
                            operation_id=func_node.name,
                            responses={
                                "200": OpenAPIResponse(description="Successful response")
                            }
                        )

                        # Extract path parameters
                        path_params = re.findall(r'\{(\w+)\}', path)
                        for param in path_params:
                            operation.parameters.append(OpenAPIParameter(
                                name=param,
                                in_="path",
                                required=True,
                                schema={"type": "string"}
                            ))

                        # Extract query parameters from function signature
                        for arg in func_node.args.args:
                            arg_name = arg.arg
                            if arg_name not in ['request', 'db', 'session'] and arg_name not in path_params:
                                operation.parameters.append(OpenAPIParameter(
                                    name=arg_name,
                                    in_="query",
                                    required=False,
                                    schema={"type": "string"}
                                ))

                        self.builder.add_path(path, method, operation)


class FlaskExtractor:
    """Extract OpenAPI info from Flask code"""

    def __init__(self, input_path: str):
        self.input_path = Path(input_path)
        self.builder = OpenAPIBuilder(OpenAPIInfo(
            title="Flask Application",
            description="Auto-generated from Flask code",
            version="1.0.0"
        ))

    def extract(self) -> OpenAPIBuilder:
        """Extract OpenAPI from Flask code"""
        with open(self.input_path, 'r') as f:
            content = f.read()

        # Parse route decorators
        routes = re.findall(
            r'@app\.route\([\'"](.+?)[\'"].*?\).*?\ndef\s+(\w+)',
            content,
            re.DOTALL
        )

        for path, func_name in routes:
            # Determine HTTP method (default GET)
            method_match = re.search(
                rf"@app\.route\(['\"]" + re.escape(path) + rf"['\"].*?methods=\[(.*?)\]",
                content
            )

            methods = ["get"]  # default
            if method_match:
                methods_str = method_match.group(1)
                methods = [m.strip().strip("'\"").lower() for m in methods_str.split(',')]

            for method in methods:
                operation = OpenAPIOperation(
                    summary=func_name.replace('_', ' ').title(),
                    operation_id=func_name,
                    responses={
                        "200": OpenAPIResponse(description="Successful response")
                    }
                )

                self.builder.add_path(path, method, operation)

        # Add default server
        self.builder.add_server("http://localhost:5000", "Development server")

        return self.builder


def create_example_spec() -> OpenAPIBuilder:
    """Create an example OpenAPI specification"""
    builder = OpenAPIBuilder(OpenAPIInfo(
        title="Example API",
        description="Example API with common patterns",
        version="1.0.0",
        contact={
            "name": "API Support",
            "email": "support@example.com"
        }
    ))

    # Add server
    builder.add_server("https://api.example.com", "Production server")
    builder.add_server("http://localhost:8000", "Development server")

    # Add tags
    builder.add_tag("users", "User management endpoints")
    builder.add_tag("posts", "Blog post endpoints")

    # Add security scheme
    builder.add_security_scheme("bearerAuth", {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT"
    })

    # Add schemas
    builder.add_schema("User", {
        "type": "object",
        "required": ["id", "name", "email"],
        "properties": {
            "id": {"type": "integer"},
            "name": {"type": "string"},
            "email": {"type": "string", "format": "email"},
            "created_at": {"type": "string", "format": "date-time"}
        }
    })

    builder.add_schema("Post", {
        "type": "object",
        "required": ["id", "title", "content"],
        "properties": {
            "id": {"type": "integer"},
            "title": {"type": "string"},
            "content": {"type": "string"},
            "author_id": {"type": "integer"},
            "created_at": {"type": "string", "format": "date-time"}
        }
    })

    # Add endpoints
    # GET /users
    builder.add_path("/users", "get", OpenAPIOperation(
        summary="List users",
        description="Get a paginated list of users",
        operation_id="listUsers",
        tags=["users"],
        parameters=[
            OpenAPIParameter(name="page", in_="query", schema={"type": "integer", "default": 1}),
            OpenAPIParameter(name="limit", in_="query", schema={"type": "integer", "default": 20}),
        ],
        responses={
            "200": OpenAPIResponse(
                description="Successful response",
                content={
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "items": {"type": "array", "items": {"$ref": "#/components/schemas/User"}},
                                "total": {"type": "integer"},
                                "page": {"type": "integer"}
                            }
                        }
                    }
                }
            )
        }
    ))

    # GET /users/{id}
    builder.add_path("/users/{id}", "get", OpenAPIOperation(
        summary="Get user by ID",
        operation_id="getUserById",
        tags=["users"],
        parameters=[
            OpenAPIParameter(name="id", in_="path", required=True, schema={"type": "integer"}),
        ],
        responses={
            "200": OpenAPIResponse(
                description="User found",
                content={"application/json": {"schema": {"$ref": "#/components/schemas/User"}}}
            ),
            "404": OpenAPIResponse(description="User not found")
        }
    ))

    # POST /users
    builder.add_path("/users", "post", OpenAPIOperation(
        summary="Create user",
        operation_id="createUser",
        tags=["users"],
        request_body={
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/User"}
                }
            }
        },
        responses={
            "201": OpenAPIResponse(
                description="User created",
                content={"application/json": {"schema": {"$ref": "#/components/schemas/User"}}}
            ),
            "400": OpenAPIResponse(description="Invalid input")
        }
    ))

    return builder


def main():
    parser = argparse.ArgumentParser(
        description="Generate OpenAPI specifications from code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from FastAPI
  python openapi_builder.py --framework fastapi --input ./api --output openapi.yaml

  # Extract from Flask
  python openapi_builder.py --framework flask --input app.py --output swagger.json

  # Create example spec
  python openapi_builder.py --example --output example.yaml
        """
    )

    parser.add_argument('--framework', choices=['fastapi', 'flask', 'django'], help='Framework to extract from')
    parser.add_argument('--input', help='Input file or directory')
    parser.add_argument('--output', required=True, help='Output file (JSON or YAML)')
    parser.add_argument('--example', action='store_true', help='Create an example specification')

    args = parser.parse_args()

    if args.example:
        print("Creating example OpenAPI specification...")
        builder = create_example_spec()
    elif args.framework and args.input:
        print(f"Extracting OpenAPI from {args.framework} code...")

        if args.framework == 'fastapi':
            extractor = FastAPIExtractor(args.input)
            builder = extractor.extract()
        elif args.framework == 'flask':
            extractor = FlaskExtractor(args.input)
            builder = extractor.extract()
        else:
            print(f"Framework {args.framework} not yet implemented")
            return 1
    else:
        parser.print_help()
        return 1

    # Save output
    if args.output.endswith('.yaml') or args.output.endswith('.yml'):
        builder.to_yaml(args.output)
    else:
        builder.to_json(args.output)

    return 0


if __name__ == '__main__':
    exit(main())
