from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import Paper
from src.schemas.paper import ArxivPaper
from src.schemas.pdf import PdfContent


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

    async def update_pdf_content(self, paper: Paper, pdf_content: PdfContent) -> Paper:
        paper.raw_text = pdf_content.raw_text
        paper.sections = [section.model_dump() for section in pdf_content.sections]
        paper.references = pdf_content.references
        paper.parser_used = pdf_content.parser_used.value
        paper.parser_metadata = pdf_content.parser_metadata
        paper.pdf_processed = True
        paper.pdf_processing_date = datetime.now(UTC)

        self.session.add(paper)
        await self.session.commit()
        await self.session.refresh(paper)

        return paper

    async def list_processed(self, limit: int = 100) -> list[Paper]:
        statement = select(Paper).where(Paper.pdf_processed.is_(True)).limit(limit)
        result = await self.session.execute(statement)

        return list(result.scalars().all())
