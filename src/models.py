from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    arxiv_id: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    abstract: Mapped[str] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(JSON)
    categories: Mapped[list[str]] = mapped_column(JSON)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    pdf_url: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
