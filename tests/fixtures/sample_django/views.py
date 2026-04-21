"""Django views (DRF ViewSets) used for extractor tests."""
from django.db.models.signals import post_save  # type: ignore[import]
from django.dispatch import receiver  # type: ignore[import]
from rest_framework import viewsets  # type: ignore[import]
from rest_framework.decorators import action  # type: ignore[import]
from rest_framework.response import Response  # type: ignore[import]

from tests.fixtures.sample_django.models import Article, Category
from tests.fixtures.sample_django.serializers import ArticleSerializer, CategorySerializer


class CategoryViewSet(viewsets.ModelViewSet):
    """CRUD ViewSet for Category."""
    queryset = Category.objects.all()
    serializer_class = CategorySerializer


class ArticleViewSet(viewsets.ModelViewSet):
    """CRUD ViewSet for Article."""
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer

    @action(detail=True, methods=["post"])
    def publish(self, request, pk=None):
        """Mark an article as published."""
        article = self.get_object()
        article.publish()
        return Response({"status": "published"})


@receiver(post_save, sender=Article)
def on_article_saved(sender, instance, created, **kwargs):
    """Signal handler: fires after every Article save."""
    if created:
        pass  # e.g. notify subscribers
