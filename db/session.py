from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from settings import SQLALCHEMY_DATABASE_URI

engine = create_engine(SQLALCHEMY_DATABASE_URI, pool_recycle=1500)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


if __name__ == "__main__":

    print('--------------------------------')
    print(f"PostgreSQL version: {1}")
