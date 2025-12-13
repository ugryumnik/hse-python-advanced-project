from pathlib import Path
from fastapi import UploadFile
import aiofiles

from infra.llm import LegalRAGAgent


class IngestionService:
    def __init__(self):
        pass

    async def processFile(self, file: UploadFile) -> None:
        pass
