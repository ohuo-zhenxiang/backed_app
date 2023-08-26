from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from db.base_class import Base
from datetime import datetime


class Task(Base):
    """
    Task model
    """
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    task_token = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    interval_seconds = Column(Integer, nullable=False)
    status = Column(String(255), default='Waiting', index=True)
    capture_path = Column(String(255), nullable=False)
    created_time = Column(DateTime, default=datetime.now().replace(microsecond=0))


class Record(Base):
    """
    Record model
    """
    __tablename__ = "records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    start_time = Column(DateTime, nullable=False)
    completed_time = Column(DateTime, nullable=False)
    face_count = Column(Integer, nullable=False)
    record_info = Column(JSON, nullable=True)
    record_image_path = Column(String(255), nullable=True)

    task_token = Column(String(255), nullable=False, index=True)
