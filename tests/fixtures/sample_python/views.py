"""Sample views file with function calls for Phase 4 testing."""
from tests.fixtures.sample_python.models import User, create_user, get_user_by_email


def list_users() -> list[User]:
    """Return all users (stub)."""
    return []


def get_user(email: str) -> User | None:
    """Look up a user. Calls get_user_by_email."""
    user = get_user_by_email(email)
    return user


def register(name: str, email: str) -> User:
    """Register a new user. Calls create_user."""
    return create_user(name=name, email=email)


class UserView:
    """View class grouping user HTTP handlers."""

    def get(self, email: str) -> User | None:
        return get_user(email)

    def post(self, name: str, email: str) -> User:
        return register(name, email)
