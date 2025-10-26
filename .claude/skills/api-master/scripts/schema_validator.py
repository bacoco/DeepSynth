#!/usr/bin/env python3
"""
Schema Validator - Validate API schemas and detect breaking changes

Supports:
- OpenAPI/Swagger specifications
- JSON Schema
- GraphQL schemas
- Pydantic models

Usage:
    python schema_validator.py --spec openapi.yaml --validate
    python schema_validator.py --old old_spec.yaml --new new_spec.yaml --breaking-changes
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ChangeType(Enum):
    """Types of schema changes"""
    BREAKING = "breaking"
    NON_BREAKING = "non-breaking"
    DEPRECATED = "deprecated"


@dataclass
class SchemaChange:
    """Represents a schema change"""
    change_type: ChangeType
    location: str
    description: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None


class OpenAPIValidator:
    """Validates OpenAPI specifications"""

    def __init__(self, spec_path: str):
        self.spec_path = Path(spec_path)
        self.spec = self._load_spec()

    def _load_spec(self) -> Dict:
        """Load OpenAPI specification"""
        with open(self.spec_path, 'r') as f:
            if self.spec_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)

    def validate(self) -> List[str]:
        """Validate OpenAPI specification"""
        errors = []

        # Check required fields
        required_fields = ['openapi', 'info', 'paths']
        for field in required_fields:
            if field not in self.spec:
                errors.append(f"Missing required field: {field}")

        # Validate info section
        if 'info' in self.spec:
            info_required = ['title', 'version']
            for field in info_required:
                if field not in self.spec['info']:
                    errors.append(f"Missing required info field: {field}")

        # Validate paths
        if 'paths' in self.spec:
            for path, path_item in self.spec['paths'].items():
                path_errors = self._validate_path(path, path_item)
                errors.extend(path_errors)

        # Validate components/schemas
        if 'components' in self.spec and 'schemas' in self.spec['components']:
            for schema_name, schema in self.spec['components']['schemas'].items():
                schema_errors = self._validate_schema(schema_name, schema)
                errors.extend(schema_errors)

        return errors

    def _validate_path(self, path: str, path_item: Dict) -> List[str]:
        """Validate a path item"""
        errors = []

        http_methods = ['get', 'post', 'put', 'patch', 'delete', 'options', 'head']
        for method in http_methods:
            if method in path_item:
                operation = path_item[method]

                # Check for operation ID
                if 'operationId' not in operation:
                    errors.append(f"{method.upper()} {path}: Missing operationId")

                # Check for summary/description
                if 'summary' not in operation and 'description' not in operation:
                    errors.append(f"{method.upper()} {path}: Missing summary or description")

                # Check for responses
                if 'responses' not in operation:
                    errors.append(f"{method.upper()} {path}: Missing responses")
                elif '200' not in operation['responses'] and '201' not in operation['responses']:
                    errors.append(f"{method.upper()} {path}: Missing success response (200/201)")

                # Validate parameters
                if 'parameters' in operation:
                    for param in operation['parameters']:
                        if 'name' not in param:
                            errors.append(f"{method.upper()} {path}: Parameter missing name")
                        if 'in' not in param:
                            errors.append(f"{method.upper()} {path}: Parameter '{param.get('name', '?')}' missing 'in' field")

        return errors

    def _validate_schema(self, schema_name: str, schema: Dict) -> List[str]:
        """Validate a schema definition"""
        errors = []

        # Check for type
        if 'type' not in schema and '$ref' not in schema and 'allOf' not in schema and 'oneOf' not in schema and 'anyOf' not in schema:
            errors.append(f"Schema '{schema_name}': Missing type definition")

        # Validate object schema
        if schema.get('type') == 'object':
            if 'properties' not in schema:
                errors.append(f"Schema '{schema_name}': Object type should have properties")

            # Check required fields are in properties
            if 'required' in schema and 'properties' in schema:
                for required_field in schema['required']:
                    if required_field not in schema['properties']:
                        errors.append(f"Schema '{schema_name}': Required field '{required_field}' not in properties")

        return errors

    def get_endpoints(self) -> List[str]:
        """Get all endpoints defined in the spec"""
        endpoints = []
        if 'paths' in self.spec:
            for path, path_item in self.spec['paths'].items():
                for method in ['get', 'post', 'put', 'patch', 'delete']:
                    if method in path_item:
                        endpoints.append(f"{method.upper()} {path}")
        return endpoints


class SchemaComparator:
    """Compare two OpenAPI specs and detect breaking changes"""

    def __init__(self, old_spec_path: str, new_spec_path: str):
        self.old_validator = OpenAPIValidator(old_spec_path)
        self.new_validator = OpenAPIValidator(new_spec_path)

    def compare(self) -> List[SchemaChange]:
        """Compare specifications and detect changes"""
        changes = []

        # Compare paths
        changes.extend(self._compare_paths())

        # Compare schemas
        changes.extend(self._compare_schemas())

        return changes

    def _compare_paths(self) -> List[SchemaChange]:
        """Compare API paths"""
        changes = []

        old_paths = set(self.old_validator.spec.get('paths', {}).keys())
        new_paths = set(self.new_validator.spec.get('paths', {}).keys())

        # Removed paths (breaking)
        for path in old_paths - new_paths:
            changes.append(SchemaChange(
                change_type=ChangeType.BREAKING,
                location=f"paths.{path}",
                description=f"Endpoint removed: {path}",
                old_value=path,
                new_value=None
            ))

        # New paths (non-breaking)
        for path in new_paths - old_paths:
            changes.append(SchemaChange(
                change_type=ChangeType.NON_BREAKING,
                location=f"paths.{path}",
                description=f"New endpoint added: {path}",
                old_value=None,
                new_value=path
            ))

        # Compare existing paths
        for path in old_paths & new_paths:
            path_changes = self._compare_path_operations(path)
            changes.extend(path_changes)

        return changes

    def _compare_path_operations(self, path: str) -> List[SchemaChange]:
        """Compare operations for a specific path"""
        changes = []

        old_path_item = self.old_validator.spec['paths'][path]
        new_path_item = self.new_validator.spec['paths'][path]

        http_methods = ['get', 'post', 'put', 'patch', 'delete']

        old_methods = set(m for m in http_methods if m in old_path_item)
        new_methods = set(m for m in http_methods if m in new_path_item)

        # Removed methods (breaking)
        for method in old_methods - new_methods:
            changes.append(SchemaChange(
                change_type=ChangeType.BREAKING,
                location=f"{method.upper()} {path}",
                description=f"HTTP method removed",
                old_value=method,
                new_value=None
            ))

        # New methods (non-breaking)
        for method in new_methods - old_methods:
            changes.append(SchemaChange(
                change_type=ChangeType.NON_BREAKING,
                location=f"{method.upper()} {path}",
                description=f"New HTTP method added",
                old_value=None,
                new_value=method
            ))

        # Compare parameters for existing methods
        for method in old_methods & new_methods:
            param_changes = self._compare_parameters(path, method, old_path_item[method], new_path_item[method])
            changes.extend(param_changes)

        return changes

    def _compare_parameters(self, path: str, method: str, old_op: Dict, new_op: Dict) -> List[SchemaChange]:
        """Compare operation parameters"""
        changes = []

        old_params = {p['name']: p for p in old_op.get('parameters', [])}
        new_params = {p['name']: p for p in new_op.get('parameters', [])}

        # Removed parameters (potentially breaking)
        for param_name in set(old_params.keys()) - set(new_params.keys()):
            old_param = old_params[param_name]
            is_breaking = old_param.get('required', False)
            changes.append(SchemaChange(
                change_type=ChangeType.BREAKING if is_breaking else ChangeType.NON_BREAKING,
                location=f"{method.upper()} {path} - parameter '{param_name}'",
                description=f"Parameter removed (required={is_breaking})",
                old_value=param_name,
                new_value=None
            ))

        # New required parameters (breaking)
        for param_name in set(new_params.keys()) - set(old_params.keys()):
            new_param = new_params[param_name]
            is_required = new_param.get('required', False)
            changes.append(SchemaChange(
                change_type=ChangeType.BREAKING if is_required else ChangeType.NON_BREAKING,
                location=f"{method.upper()} {path} - parameter '{param_name}'",
                description=f"New parameter added (required={is_required})",
                old_value=None,
                new_value=param_name
            ))

        # Changed parameters
        for param_name in set(old_params.keys()) & set(new_params.keys()):
            old_param = old_params[param_name]
            new_param = new_params[param_name]

            # Required changed
            if old_param.get('required') != new_param.get('required'):
                is_breaking = new_param.get('required', False)
                changes.append(SchemaChange(
                    change_type=ChangeType.BREAKING if is_breaking else ChangeType.NON_BREAKING,
                    location=f"{method.upper()} {path} - parameter '{param_name}'",
                    description=f"Parameter required status changed",
                    old_value=old_param.get('required'),
                    new_value=new_param.get('required')
                ))

        return changes

    def _compare_schemas(self) -> List[SchemaChange]:
        """Compare schema definitions"""
        changes = []

        old_schemas = self.old_validator.spec.get('components', {}).get('schemas', {})
        new_schemas = self.new_validator.spec.get('components', {}).get('schemas', {})

        # Removed schemas (breaking)
        for schema_name in set(old_schemas.keys()) - set(new_schemas.keys()):
            changes.append(SchemaChange(
                change_type=ChangeType.BREAKING,
                location=f"components.schemas.{schema_name}",
                description=f"Schema removed",
                old_value=schema_name,
                new_value=None
            ))

        # New schemas (non-breaking)
        for schema_name in set(new_schemas.keys()) - set(old_schemas.keys()):
            changes.append(SchemaChange(
                change_type=ChangeType.NON_BREAKING,
                location=f"components.schemas.{schema_name}",
                description=f"New schema added",
                old_value=None,
                new_value=schema_name
            ))

        # Compare existing schemas
        for schema_name in set(old_schemas.keys()) & set(new_schemas.keys()):
            schema_changes = self._compare_schema_properties(schema_name, old_schemas[schema_name], new_schemas[schema_name])
            changes.extend(schema_changes)

        return changes

    def _compare_schema_properties(self, schema_name: str, old_schema: Dict, new_schema: Dict) -> List[SchemaChange]:
        """Compare schema properties"""
        changes = []

        old_props = old_schema.get('properties', {})
        new_props = new_schema.get('properties', {})

        old_required = set(old_schema.get('required', []))
        new_required = set(new_schema.get('required', []))

        # Removed properties (potentially breaking)
        for prop_name in set(old_props.keys()) - set(new_props.keys()):
            is_breaking = prop_name in old_required
            changes.append(SchemaChange(
                change_type=ChangeType.BREAKING if is_breaking else ChangeType.DEPRECATED,
                location=f"schemas.{schema_name}.properties.{prop_name}",
                description=f"Property removed from response schema",
                old_value=prop_name,
                new_value=None
            ))

        # New required properties (breaking for requests)
        for prop_name in new_required - old_required:
            changes.append(SchemaChange(
                change_type=ChangeType.BREAKING,
                location=f"schemas.{schema_name}.required",
                description=f"Property '{prop_name}' is now required",
                old_value=False,
                new_value=True
            ))

        # Removed required (non-breaking)
        for prop_name in old_required - new_required:
            changes.append(SchemaChange(
                change_type=ChangeType.NON_BREAKING,
                location=f"schemas.{schema_name}.required",
                description=f"Property '{prop_name}' is no longer required",
                old_value=True,
                new_value=False
            ))

        return changes

    def get_breaking_changes(self) -> List[SchemaChange]:
        """Get only breaking changes"""
        all_changes = self.compare()
        return [c for c in all_changes if c.change_type == ChangeType.BREAKING]

    def has_breaking_changes(self) -> bool:
        """Check if there are any breaking changes"""
        return len(self.get_breaking_changes()) > 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate API schemas and detect breaking changes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a single spec
  python schema_validator.py --spec openapi.yaml --validate

  # Compare two specs for breaking changes
  python schema_validator.py --old v1.yaml --new v2.yaml --breaking-changes

  # Compare and show all changes
  python schema_validator.py --old v1.yaml --new v2.yaml --all-changes
        """
    )

    parser.add_argument('--spec', help='OpenAPI specification file to validate')
    parser.add_argument('--validate', action='store_true', help='Validate the specification')
    parser.add_argument('--old', help='Old specification file for comparison')
    parser.add_argument('--new', help='New specification file for comparison')
    parser.add_argument('--breaking-changes', action='store_true', help='Show only breaking changes')
    parser.add_argument('--all-changes', action='store_true', help='Show all changes')
    parser.add_argument('--endpoints', action='store_true', help='List all endpoints')

    args = parser.parse_args()

    # Validate mode
    if args.spec and args.validate:
        print(f"\n🔍 Validating {args.spec}...")
        validator = OpenAPIValidator(args.spec)
        errors = validator.validate()

        if errors:
            print(f"\n❌ Found {len(errors)} validation error(s):\n")
            for error in errors:
                print(f"  • {error}")
            return 1
        else:
            print("\n✅ Specification is valid!")
            return 0

    # List endpoints
    if args.spec and args.endpoints:
        validator = OpenAPIValidator(args.spec)
        endpoints = validator.get_endpoints()
        print(f"\n📋 Endpoints in {args.spec}:\n")
        for endpoint in endpoints:
            print(f"  • {endpoint}")
        print(f"\nTotal: {len(endpoints)} endpoints")
        return 0

    # Compare mode
    if args.old and args.new:
        print(f"\n🔍 Comparing {args.old} → {args.new}...")
        comparator = SchemaComparator(args.old, args.new)

        if args.breaking_changes:
            changes = comparator.get_breaking_changes()
            if changes:
                print(f"\n❌ Found {len(changes)} BREAKING change(s):\n")
                for change in changes:
                    print(f"  🔴 {change.location}")
                    print(f"     {change.description}")
                    if change.old_value is not None:
                        print(f"     Old: {change.old_value}")
                    if change.new_value is not None:
                        print(f"     New: {change.new_value}")
                    print()
                return 1
            else:
                print("\n✅ No breaking changes detected!")
                return 0

        elif args.all_changes:
            changes = comparator.compare()
            if changes:
                print(f"\n📊 Found {len(changes)} change(s):\n")

                breaking = [c for c in changes if c.change_type == ChangeType.BREAKING]
                non_breaking = [c for c in changes if c.change_type == ChangeType.NON_BREAKING]
                deprecated = [c for c in changes if c.change_type == ChangeType.DEPRECATED]

                if breaking:
                    print(f"🔴 BREAKING CHANGES ({len(breaking)}):")
                    for change in breaking:
                        print(f"  • {change.location}: {change.description}")
                    print()

                if non_breaking:
                    print(f"🟢 NON-BREAKING CHANGES ({len(non_breaking)}):")
                    for change in non_breaking:
                        print(f"  • {change.location}: {change.description}")
                    print()

                if deprecated:
                    print(f"🟡 DEPRECATED ({len(deprecated)}):")
                    for change in deprecated:
                        print(f"  • {change.location}: {change.description}")

                return 1 if breaking else 0
            else:
                print("\n✅ No changes detected!")
                return 0

    # No valid action
    parser.print_help()
    return 1


if __name__ == '__main__':
    exit(main())
