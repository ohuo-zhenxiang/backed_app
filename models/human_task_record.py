from sqlalchemy import Column, Integer, String, DateTime, JSON, ARRAY
from datetime import datetime
from db.base_class import Base


class HumanTask(Base):
    """
    HumanTask Model
    """
    __tablename__ = "human_tasks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    expand_tasks = Column(ARRAY(String(250)), default=[])
    task_token = Column(String(255), nullable=False, index=True)
    task_name = Column(String(255), nullable=False, index=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    interval_seconds = Column(Integer, nullable=False)
    status = Column(String(255), default="Waiting", index=True)
    capture_path = Column(String(255), nullable=False)
    created_time = Column(DateTime, default=datetime.now().replace(microsecond=0))


class HumanRecord(Base):
    """
    HumanRecord Model
    """
    __tablename__ = "human_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    start_time = Column(DateTime, nullable=False)
    completed_time = Column(DateTime, nullable=False)
    human_count = Column(Integer, nullable=False)
    record_info = Column(JSON, nullable=True)
    record_image_path = Column(String(255), nullable=True)

    task_token = Column(String(255), nullable=False, index=True)
