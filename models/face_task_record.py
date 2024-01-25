from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, ARRAY
from sqlalchemy.orm import relationship
from db.base_class import Base
from datetime import datetime


class Task(Base):
    """
    Task model
    """
    __tablename__ = "face_tasks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ex_detect = Column(ARRAY(String(250)), default=[])
    task_token = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    interval_seconds = Column(Integer, nullable=False)
    status = Column(String(255), default='Waiting', index=True)
    associated_group_id = Column(Integer, nullable=False)
    capture_path = Column(String(255), nullable=False)
    created_time = Column(DateTime, default=datetime.now().replace(microsecond=0))


class Record(Base):
    """
    Record model
    """
    __tablename__ = "face_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    start_time = Column(DateTime, nullable=False)
    completed_time = Column(DateTime, nullable=False)
    face_count = Column(Integer, nullable=False)
    record_info = Column(JSON, nullable=True)
    record_image_path = Column(String(255), nullable=True)
    record_names = Column(ARRAY(String), nullable=False)

    task_token = Column(String(255), nullable=False, index=True)

