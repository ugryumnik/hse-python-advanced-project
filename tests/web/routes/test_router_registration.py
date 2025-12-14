from fastapi.routing import APIRoute

from src.web.routes import router


def test_router_includes_expected_routes():
    paths = {route.path for route in router.routes if isinstance(route, APIRoute)}
    assert "/ask" in paths
    assert "/upload" in paths
    assert "/source" in paths
    assert "/generate" in paths
