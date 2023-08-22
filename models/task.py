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
    name = Column(String(255), nullable=False, index=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    interval_seconds = Column(Integer, nullable=False)
    status = Column(String(255), default='waiting', index=True)
    created_time = Column(DateTime, default=datetime.now)

    records = relationship("Record", back_populates="task")


class Record(Base):
    """
    Record model
    """
    __tablename__ = "records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    execution_time = Column(DateTime, nullable=False)
    completed_time = Column(DateTime, nullable=False)
    record_info = Column(JSON, nullable=False)

    task_id = Column(Integer, ForeignKey("tasks.id"))
    task = relationship("Task", back_populates="records")