"""
REST API Endpoint Template (FastAPI)

Replace {RESOURCE} with your resource name (e.g., User, Post, Product)
Replace {resource} with lowercase version (e.g., user, post, product)
Replace {resources} with plural version (e.g., users, posts, products)
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from database import get_db
from schemas.{resource} import (
    {RESOURCE}Create,
    {RESOURCE}Update,
    {RESOURCE}Response,
    {RESOURCE}List
)
from crud.{resource} import {resource}_crud

router = APIRouter(
    prefix="/{resources}",
    tags=["{resources}"],
)


@router.post(
    "/",
    response_model={RESOURCE}Response,
    status_code=status.HTTP_201_CREATED,
    summary="Create a {resource}",
    description="Create a new {resource} with the provided data."
)
async def create_{resource}(
    {resource}: {RESOURCE}Create,
    db: Session = Depends(get_db)
):
    """
    Create a new {resource}.

    - **{resource}**: {RESOURCE} data to create
    - Returns: Created {resource} object
    """
    try:
        return {resource}_crud.create(db=db, obj_in={resource})
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create {resource}: {{str(e)}}"
        )


@router.get(
    "/",
    response_model={RESOURCE}List,
    summary="List {resources}",
    description="Retrieve a paginated list of {resources} with optional filtering."
)
async def list_{resources}(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of items to return"),
    # Add your filter parameters here
    # search: Optional[str] = Query(None, description="Search term"),
    # sort_by: str = Query("created_at", description="Sort field"),
    db: Session = Depends(get_db)
):
    """
    List {resources} with pagination.

    - **skip**: Pagination offset (default: 0)
    - **limit**: Page size (default: 20, max: 100)
    - Returns: Paginated list of {resources}
    """
    try:
        items = {resource}_crud.get_multi(db=db, skip=skip, limit=limit)
        total = {resource}_crud.count(db=db)

        return {{
            "items": items,
            "total": total,
            "page": skip // limit + 1,
            "page_size": limit,
            "pages": (total + limit - 1) // limit
        }}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch {resources}: {{str(e)}}"
        )


@router.get(
    "/{{id}}",
    response_model={RESOURCE}Response,
    summary="Get {resource} by ID",
    description="Retrieve a specific {resource} by its ID."
)
async def get_{resource}(
    id: int,
    db: Session = Depends(get_db)
):
    """
    Get {resource} by ID.

    - **id**: {RESOURCE} ID
    - Returns: {RESOURCE} object
    - Raises 404 if not found
    """
    {resource} = {resource}_crud.get(db=db, id=id)
    if not {resource}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{RESOURCE} with id {{id}} not found"
        )
    return {resource}


@router.put(
    "/{{id}}",
    response_model={RESOURCE}Response,
    summary="Update {resource}",
    description="Update an existing {resource}."
)
async def update_{resource}(
    id: int,
    {resource}_update: {RESOURCE}Update,
    db: Session = Depends(get_db)
):
    """
    Update {resource}.

    - **id**: {RESOURCE} ID
    - **{resource}_update**: Fields to update
    - Returns: Updated {resource} object
    - Raises 404 if not found
    """
    {resource} = {resource}_crud.get(db=db, id=id)
    if not {resource}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{RESOURCE} with id {{id}} not found"
        )

    try:
        return {resource}_crud.update(
            db=db,
            db_obj={resource},
            obj_in={resource}_update
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update {resource}: {{str(e)}}"
        )


@router.delete(
    "/{{id}}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete {resource}",
    description="Delete a {resource} by ID."
)
async def delete_{resource}(
    id: int,
    db: Session = Depends(get_db)
):
    """
    Delete {resource}.

    - **id**: {RESOURCE} ID
    - Returns: No content (204)
    - Raises 404 if not found
    """
    {resource} = {resource}_crud.get(db=db, id=id)
    if not {resource}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{RESOURCE} with id {{id}} not found"
        )

    try:
        {resource}_crud.remove(db=db, id=id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete {resource}: {{str(e)}}"
        )
