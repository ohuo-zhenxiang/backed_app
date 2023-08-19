from datetime import datetime
from sqlalchemy import Boolean, Column, Integer, String, LargeBinary, DateTime
from sqlalchemy.orm import relationship
from db.base_class import Base


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    full_name = Column(String, index=True)
    phone = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_activate = Column(Boolean(), default=True)
    is_superuser = Column(Boolean(), default=False)
