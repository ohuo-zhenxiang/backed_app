from datetime import datetime
from sqlalchemy import Table, Column, Integer, String, DateTime, LargeBinary, ForeignKey
from sqlalchemy.orm import relationship
from db.base_class import Base

# 定义中间表用于face和group之间多对多的关系
face_group_association = Table(
    "face_group_association",
    Base.metadata,
    Column("face_id", Integer, ForeignKey("faces.id")),
    Column("group_id", Integer, ForeignKey("groups.id")),
)


class Face(Base):
    __tablename__ = "faces"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    name = Column(String(255), nullable=False)
    phone = Column(String(255), unique=True, index=True, nullable=False)
    gender = Column(String(50), nullable=True)
    face_image_path = Column(String(255), nullable=False)
    face_features = Column(LargeBinary, nullable=False)
    created_time = Column(DateTime, default=datetime.now().replace(microsecond=0))

    group = relationship("Group", secondary=face_group_association, back_populates="members")
