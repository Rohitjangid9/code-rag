"""DRF serializers for Django extractor tests."""
from rest_framework import serializers  # type: ignore[import]

from tests.fixtures.sample_django.models import Article, Category


class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ["id", "name", "slug", "created_at"]


class ArticleSerializer(serializers.ModelSerializer):
    category = CategorySerializer(read_only=True)
    category_id = serializers.PrimaryKeyRelatedField(
        queryset=Category.objects.all(), source="category", write_only=True
    )

    class Meta:
        model = Article
        fields = ["id", "title", "body", "category", "category_id", "published", "created_at"]
