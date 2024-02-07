from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from settings import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_ASYNC_DATABASE_URI

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import AsyncAdaptedQueuePool

engine = create_engine(SQLALCHEMY_DATABASE_URI, pool_recycle=1500)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


async_engine = create_async_engine(SQLALCHEMY_ASYNC_DATABASE_URI,
                                   echo=False,  # is print log
                                   echo_pool=True,
                                   pool_recycle=1500,
                                   poolclass=AsyncAdaptedQueuePool,
                                   )
AsyncSessionLocal = async_sessionmaker(async_engine, class_=AsyncSession, autocommit=False, autoflush=False)


if __name__ == "__main__":
    async def main():
        async with async_engine.connect() as connection:
            query = text("SELECT version()")
            result = await connection.execute(query)
            version = result.scalar()
            print(f"PostgreSQL version: {version}")

    import asyncio
    asyncio.run(main())
