from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/demo"

# 创建一个SQLAlchemy的“引擎”,
# 用SQLite数据库的话需要加参数 connect_args={"check_same_thread": False}
# engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# 每个SessionLocal的实例都是一个数据库会话，命名SessionLocal区别于Session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()