"""Pydantic request/response models used for FastAPI extractor tests."""
from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    """Request body for creating a user."""
    name: str
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Response body for a user."""
    id: int
    name: str
    email: str

    model_config = {"from_attributes": True}


class ArticleCreate(BaseModel):
    title: str
    body: str
    category_id: int


class ArticleResponse(BaseModel):
    id: int
    title: str
    body: str
    published: bool
