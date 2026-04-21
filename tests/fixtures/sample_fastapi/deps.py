"""FastAPI dependency injection functions."""
from typing import Generator

from fastapi import HTTPException, status  # type: ignore[import]


def get_db() -> Generator:
    """Yield a database session (stub)."""
    db = object()  # placeholder for Session
    try:
        yield db
    finally:
        pass


def get_current_user(token: str = ""):
    """Validate Bearer token and return current user (stub)."""
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return {"id": 1, "name": "admin"}
