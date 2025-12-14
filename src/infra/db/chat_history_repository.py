from typing import Optional, List

from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ChatHistory


class ChatHistoryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        user_id: int,
        query: str,
        answer: str,
        used_sources: Optional[dict] = None,
    ) -> ChatHistory:
        message = ChatHistory(
            user_id=user_id,
            query=query,
            answer=answer,
            used_sources=used_sources,
        )
        self.session.add(message)
        await self.session.commit()
        await self.session.refresh(message)
        return message

    async def get_by_id(self, message_id: int) -> Optional[ChatHistory]:
        result = await self.session.execute(
            select(ChatHistory).where(ChatHistory.id == message_id)
        )
        return result.scalar_one_or_none()

    async def get_by_user(
        self,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ChatHistory]:
        result = await self.session.execute(
            select(ChatHistory)
            .where(ChatHistory.user_id == user_id)
            .order_by(ChatHistory.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_recent(self, user_id: int, limit: int = 10) -> List[ChatHistory]:
        result = await self.session.execute(
            select(ChatHistory)
            .where(ChatHistory.user_id == user_id)
            .order_by(ChatHistory.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_by_user(self, user_id: int) -> int:
        result = await self.session.execute(
            select(func.count(ChatHistory.id))
            .where(ChatHistory.user_id == user_id)
        )
        return result.scalar() or 0

    async def delete_by_user(self, user_id: int) -> int:
        result = await self.session.execute(
            delete(ChatHistory).where(ChatHistory.user_id == user_id)
        )
        await self.session.commit()
        return result.rowcount