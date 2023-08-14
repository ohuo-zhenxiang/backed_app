from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, LargeBinary
from sqlalchemy.orm import relationship
from db.base_class import Base


class Face(Base):
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    name = Column(String(255), nullable=False)
    phone = Column(String(255), unique=True, index=True, nullable=False)
    gender = Column(String(50), nullable=True)
    face_image_path = Column(String(255), nullable=False)
    face_features = Column(LargeBinary, nullable=False)
    created_time = Column(DateTime, default=datetime.now)

