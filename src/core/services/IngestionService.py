from pathlib import Path
from fastapi import UploadFile
import aiofiles

from infra.llm import LegalRAGAgent


class IngestionService:
    def __init__(self, agent: LegalRAGAgent | None = None):
        self.agent = agent

    async def processFile(self, file: UploadFile) -> None:
        """Process uploaded file and ingest into vector store.

        Minimal implementation placeholder — real logic lives elsewhere.
        """
        return None
