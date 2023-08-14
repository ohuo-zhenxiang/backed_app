from typing import Any, Dict, Optional, Union, List, Type
from sqlalchemy.orm import Session
from crud.crud_base import CRUDBase
from models import Face
from models.face import Face
from schemas.face import FaceCreate, FaceUpdate, FaceSelect


class CRUDFace(CRUDBase[Face, FaceCreate, FaceUpdate]):
    def create(self, db: Session, *, obj_in: FaceCreate) -> Face:
        db_obj = Face(name=obj_in.name, phone=obj_in.phone, gender=obj_in.gender,
                      face_image_path=obj_in.face_image_path, face_features=obj_in.face_features)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_by_phone(self, db: Session, *, phone: str) -> Optional[Face]:
        return db.query(Face).filter(Face.phone == phone).first()

    def get_faces(self, db: Session) -> list[Type[Face]]:
        return db.query(Face).all()


crud_face = CRUDFace(Face)
