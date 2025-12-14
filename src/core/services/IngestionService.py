from pathlib import Path
from fastapi import UploadFile
import aiofiles

from infra.llm import LegalRAGAgent


class IngestionService:
    def __init__(self, agent: LegalRAGAgent | None = None):
        self.agent = agent

    async def processFile(self, file: UploadFile) -> int:
        """Save uploaded file to disk, index it, and return number of chunks."""
        import tempfile
        import os
        if not self.agent:
            raise RuntimeError("LegalRAGAgent not initialized")

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)

        try:
            async with aiofiles.open(temp_path, 'wb') as out_file:
                while True:
                    chunk = await file.read(8192)
                    if not chunk:
                        break
                    await out_file.write(chunk)

            chunks_count = await self.agent.add_document(temp_path)
            return chunks_count
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
