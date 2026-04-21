"""Django URL configuration used for extractor tests."""
from django.urls import include, path  # type: ignore[import]
from rest_framework.routers import DefaultRouter  # type: ignore[import]

from tests.fixtures.sample_django.views import ArticleViewSet, CategoryViewSet

router = DefaultRouter()
router.register(r"categories", CategoryViewSet, basename="category")
router.register(r"articles", ArticleViewSet, basename="article")

urlpatterns = [
    path("api/v1/", include(router.urls)),
    path("api/v1/health/", lambda r: None, name="health"),
]
