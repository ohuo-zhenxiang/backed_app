from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from db.base_class import Base


class Camera(Base):
    """
    Camera model
    """
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    cam_token = Column(String(255), nullable=False, index=True)
    cam_name = Column(String(255), nullable=False, index=True)
    cam_type = Column(String(255), nullable=False, index=True)
    cam_url = Column(String(255), nullable=False, index=True)
    cam_status = Column(Boolean, nullable=False, index=True)
    update_time = Column(DateTime, default=datetime.now().replace(microsecond=0))
