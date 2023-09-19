"""
为避免SQLAlchemy模型和Pydantic模型之间的混淆，
models.py ---- SQLAlchemy模型文件
schemas.py ---- Pydantic模型文件
"""
from sqlalchemy import Column, Boolean, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from DEMO.sql_demo.database import Base


class User(Base):
    __tablename__ = "demo_users"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

    # 一个user可能有多个item，访问User.items就返回该user的所有关联的items
    items = relationship("Item", back_populates="owner")


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    title = Column(String, index=True)
    description = Column(String, index=True)
    # user对item是一对多关系，所以需要一个外键，当访问owner_id时获取users表中的记录
    owner_id = Column(Integer, ForeignKey('demo_users.id'))

    owner = relationship("User", back_populates="items")
