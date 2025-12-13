from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, Form
from pydantic import BaseModel

from infra.llm.agent import LegalRAGAgent


class UploadResponse(BaseModel):
    message: str
    chunks_added: int


upload_router = APIRouter()


@upload_router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile,
    user_id: int = Form(...)
):
    try:
        junk_dir = Path("junk")
        junk_dir.mkdir(exist_ok=True)
        file_path = junk_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())

        return UploadResponse(
            message=f"Document '{file.filename}' uploaded to junk",
            chunks_added=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
