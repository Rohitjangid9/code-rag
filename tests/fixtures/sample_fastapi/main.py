"""FastAPI application fixture for extractor tests."""
from fastapi import APIRouter, Depends, FastAPI, status  # type: ignore[import]

from tests.fixtures.sample_fastapi.deps import get_current_user, get_db
from tests.fixtures.sample_fastapi.models import ArticleCreate, ArticleResponse, UserCreate, UserResponse

app = FastAPI(title="Sample API", version="1.0.0")

# ── user router ────────────────────────────────────────────────────────────────
users_router = APIRouter(prefix="/users", tags=["users"])


@users_router.get("/", response_model=list[UserResponse], status_code=status.HTTP_200_OK)
def list_users(db=Depends(get_db)):
    """Return all users."""
    return []


@users_router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db=Depends(get_db)):
    """Return a single user by ID."""
    return {"id": user_id, "name": "alice", "email": "alice@example.com"}


@users_router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db=Depends(get_db), current=Depends(get_current_user)):
    """Create a new user (requires auth)."""
    return {"id": 99, **user.model_dump(exclude={"password"})}


# ── article router ─────────────────────────────────────────────────────────────
articles_router = APIRouter(prefix="/articles", tags=["articles"])


@articles_router.get("/", response_model=list[ArticleResponse])
def list_articles(db=Depends(get_db)):
    """Return all articles."""
    return []


@articles_router.post("/", response_model=ArticleResponse, status_code=201)
def create_article(article: ArticleCreate, db=Depends(get_db)):
    """Create a new article."""
    return {"id": 1, "title": article.title, "body": article.body, "published": False}


# ── health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
def health():
    """Liveness probe."""
    return {"status": "ok"}


app.include_router(users_router, prefix="/api/v1")
app.include_router(articles_router, prefix="/api/v1")
