from datetime import datetime
from sqlalchemy.orm import Session
from crud.crud_base import CRUDBase
from models import Camera
from schemas import CameraCreate, CameraUpdate


class CRUDCamera(CRUDBase[Camera, CameraCreate, CameraUpdate]):
    def create_camera(self, db: Session, obj_in: CameraCreate) -> Camera:
        db_obj = Camera(**obj_in)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def delete_camera(self, db: Session, id: int):
        db.query(Camera).filter(Camera.id == id).delete()
        db.commit()
        return

    def update_camera_status(self, db: Session, id: int, cam_status: bool):
        db_cam = db.query(Camera).filter(Camera.id == id).first()
        db_cam.cam_status = cam_status
        db_cam.update_time = datetime.now().replace(microsecond=0)
        db.commit()
        # print('db do?', cam_status)
        return


crud_camera = CRUDCamera(Camera)
