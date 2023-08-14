from datetime import datetime
from sqlalchemy import Boolean, Column, Integer, String, LargeBinary, DateTime
from sqlalchemy.orm import relationship
from db.base_class import Base


class User(Base):
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    full_name = Column(String, index=True)
    phone = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_activate = Column(Boolean(), default=True)
    is_superuser = Column(Boolean(), default=False)


# class Face(Base):
#     id = Column(Integer, primary_key=True, autoincrement=True, index=True)
#     name = Column(String(255), nullable=False)
#     phone = Column(String(255), unique=True, index=True, nullable=False)
#     gender = Column(String(50), nullable=True)
#     face_image_path = Column(String(255), nullable=False)
#     face_features = Column(LargeBinary, nullable=False)
#     created_time = Column(DateTime, default=datetime.now)
