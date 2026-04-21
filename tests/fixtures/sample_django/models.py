"""Django ORM models used for extractor tests."""
from django.db import models  # type: ignore[import]


class Category(models.Model):
    """A content category."""
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class Article(models.Model):
    """A published article belonging to a category."""
    title = models.CharField(max_length=255)
    body = models.TextField()
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name="articles")
    published = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def publish(self) -> None:
        """Mark the article as published."""
        self.published = True
        self.save()

    def __str__(self) -> str:
        return self.title
