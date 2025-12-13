from .routes import router
from fastapi import FastAPI

app = FastAPI(title="Legal RAG Bot API", version="1.0.0")

app.include_router(router)
