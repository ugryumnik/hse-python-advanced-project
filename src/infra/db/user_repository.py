from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import User


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_telegram_id(self, tg_id: int) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.telegram_user_id == tg_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, user_id: int) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        tg_id: int,
        role: str,
        username: Optional[str] = None,
    ) -> User:
        user = await self.get_by_telegram_id(tg_id)
        if user is None:
            user = User(
                telegram_user_id=tg_id,
                role=role,
                username=username,
            )
            self.session.add(user)
        else:
            user.role = role
            if username is not None:
                user.username = username
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def delete_by_telegram_id(self, tg_id: int) -> bool:
        user = await self.get_by_telegram_id(tg_id)
        if user:
            await self.session.delete(user)
            await self.session.commit()
            return True
        return False