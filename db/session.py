from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from settings import SQLALCHEMY_DATABASE_URI

engine = create_engine(SQLALCHEMY_DATABASE_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

if __name__ == "__main__":
    with engine.connect() as connection:
        query = text("SELECT version();")
        result = connection.execute(query)
        version = result.scalar()

    print('--------------------------------')
    print(f"PostgreSQL version: {version}")
