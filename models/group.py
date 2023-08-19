from .face import face_group_association
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from db.base_class import Base


class Group(Base):
    """
    Group model class
    """
    __tablename__ = 'groups'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(String(255), nullable=True)

    members = relationship("Face", secondary=face_group_association, back_populates="group")

