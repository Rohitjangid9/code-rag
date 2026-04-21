"""Sample Django-like models for testing the symbol extractor."""


class User(object):
    """A user in the system."""

    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email

    def full_name(self) -> str:
        """Return the display name."""
        return self.name

    def is_admin(self) -> bool:
        return False


class AdminUser(User):
    """Admin variant of User."""

    def is_admin(self) -> bool:
        return True

    def deactivate(self, reason: str) -> None:
        """Deactivate the admin account."""
        pass


def get_user_by_email(email: str) -> User | None:
    """Fetch a user by their email address."""
    return None


def create_user(name: str, email: str) -> User:
    """Create and persist a new user."""
    return User(name=name, email=email)
