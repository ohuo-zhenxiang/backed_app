import os
from typing import Any, Dict, Optional, Union, List, Type
from sqlalchemy import desc
from sqlalchemy.orm import Session
from crud.crud_base import CRUDBase
from datetime import datetime
from models.face import Face
from schemas.face import FaceCreate, FaceUpdate, FaceSelect


class CRUDFace(CRUDBase[Face, FaceCreate, FaceUpdate]):
    def create_face(self, db: Session, *, obj_in: FaceCreate) -> Face:
        db_obj = Face(name=obj_in.name, phone=obj_in.phone, gender=obj_in.gender,
                      face_image_path=obj_in.face_image_path, face_features=obj_in.face_features,
                      source=obj_in.source, created_time=datetime.now().replace(microsecond=0))

        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_faces(self, db: Session) -> list[Type[Face]]:
        return db.query(Face).order_by(desc(Face.id)).all()

    def get_by_phone(self, db: Session, *, phone: str) -> Optional[Face]:
        return db.query(Face).filter(Face.phone == phone).first()

    def get_face_by_id(self, db: Session, *, id: int) -> Optional[Face]:
        return db.query(Face).filter(Face.id == id).first()

    def delete_face_by_id(self, db: Session, *, id: int) -> bool:
        db_obj = self.get_face_by_id(db, id=id)
        os.remove(db_obj.face_image_path)
        db_obj.group.clear()
        db.delete(db_obj)
        db.commit()
        return True

    def update_face_by_id(self, db: Session, *, name: str, phone: str, face_id: int, gender: str = None) -> bool:
        if db.query(Face).filter(Face.phone == phone, Face.id != face_id).first():
            return False
        db_face = db.query(Face).filter(Face.id == face_id).first()
        db_face.name = name
        db_face.phone = phone
        db_face.gender = gender
        db.commit()
        return True


crud_face = CRUDFace(Face)
