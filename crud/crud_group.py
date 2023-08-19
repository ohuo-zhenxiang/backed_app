from typing import Any, Optional, List, Type

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from crud.crud_user import CRUDBase
from models import Group, Face
from schemas import GroupCreate, GroupUpdate


class CRUDGroup(CRUDBase[Group, GroupCreate, GroupUpdate]):
    def create_group(self, db: Session, name: str, description: str) -> Group:
        db_obj = Group(name=name, description=description)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_group_by_id(self, db: Session, *, id: int) -> Optional[Group]:
        return db.query(Group).filter(Group.id == id).first()

    def remove_group_by_id(self, db: Session, *, id: int) -> Optional[Group]:
        db_obj = db.query(Group).filter(Group.id == id).first()
        db.delete(db_obj)
        db.commit()
        return db_obj

    def update_group_by_id(self, db: Session, *, name: str, desc: str, group_id: int) -> Group:
        db_group = db.query(Group).filter(Group.id == group_id).first()
        if not db_group:
            return False
        else:
            db_group.name = name
            db_group.description = desc
            db.commit()
            return True

    def get_by_name(self, db: Session, name: str) -> Optional[Group]:
        return db.query(Group).filter(Group.name == name).first()

    def get_groups(self, db: Session, *, skip: int = 0, limit: int = 100) -> List[Group]:
        return db.query(Group).offset(skip).limit(limit).all()

    """组员的CRUD"""

    def add_group_members(self, db: Session, *, group_id: int, member_ids: list[int]):
        group = db.query(Group).filter(Group.id == group_id).first()
        if group:
            faces = db.query(Face).filter(Face.id.in_(member_ids)).all()
            group.members.extend(faces)
            db.commit()
            return True
        return False

    def remove_group_members(self, db: Session, *, group_id: int, member_ids: list[int]):
        group = db.query(Group).filter(Group.id == group_id).first()
        if group:
            faces = db.query(Face).filter(Face.id.in_(member_ids)).all()
            group.members.remove(*faces)
            db.commit()
            return True
        return False

    def get_group_members(self, db: Session, group_id: int):
        group = db.query(Group).filter(Group.id == group_id).first()
        if group:
            return group.members
        return []


crud_group = CRUDGroup(Group)
