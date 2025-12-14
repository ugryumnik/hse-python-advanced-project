from fastapi import APIRouter

from .ask import ask_router
from .upload import upload_router
from .source import source_router

router = APIRouter()
router.include_router(ask_router)
router.include_router(upload_router)
router.include_router(source_router)
