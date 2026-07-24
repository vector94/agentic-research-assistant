import asyncio

from sqlalchemy import text

from src.database import engine


async def main() -> None:
    async with engine.connect() as connection:
        result = await connection.execute(text("SELECT version()"))
        print(result.scalar_one())

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
