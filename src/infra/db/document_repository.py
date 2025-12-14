from typing import Optional, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Document


class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        filename: str,
        file_path: str,
        file_hash: str,
        uploaded_by: Optional[int] = None,
        status: str = "processing",
    ) -> Document:
        document = Document(
            filename=filename,
            file_path=file_path,
            file_hash=file_hash,
            uploaded_by=uploaded_by,
            status=status,
        )
        self.session.add(document)
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def get_by_id(self, doc_id: int) -> Optional[Document]:
        result = await self.session.execute(
            select(Document).where(Document.id == doc_id)
        )
        return result.scalar_one_or_none()

    async def get_by_hash(self, file_hash: str) -> Optional[Document]:
        result = await self.session.execute(
            select(Document).where(Document.file_hash == file_hash)
        )
        return result.scalar_one_or_none()

    async def get_by_user(self, user_id: int) -> List[Document]:
        result = await self.session.execute(
            select(Document).where(Document.uploaded_by == user_id)
        )
        return list(result.scalars().all())

    async def get_all(self, limit: int = 100, offset: int = 0) -> List[Document]:
        result = await self.session.execute(
            select(Document)
            .order_by(Document.uploaded_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def update_status(
        self, doc_id: int, status: str
    ) -> Optional[Document]:
        document = await self.get_by_id(doc_id)
        if document:
            document.status = status
            await self.session.commit()
            await self.session.refresh(document)
        return document

    async def delete(self, doc_id: int) -> bool:
        document = await self.get_by_id(doc_id)
        if document:
            await self.session.delete(document)
            await self.session.commit()
            return True
        return False