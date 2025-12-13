from sqlalchemy import BigInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, index=True, unique=True)
    role: Mapped[str] = mapped_column(String(16))  # "admin" or "user"
