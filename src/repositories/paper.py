from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Paper
from src.schemas.paper import ArxivPaper


class PaperRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_arxiv_id(self, arxiv_id: str) -> Paper | None:
        statement = select(Paper).where(Paper.arxiv_id == arxiv_id)
        result = await self.session.execute(statement)

        return result.scalar_one_or_none()

    async def add(self, paper: ArxivPaper) -> Paper:
        database_paper = Paper(**paper.model_dump())

        self.session.add(database_paper)
        await self.session.commit()
        await self.session.refresh(database_paper)

        return database_paper
