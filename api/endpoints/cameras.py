import uuid
from typing import Any, List

from fastapi import APIRouter, Depends, Form
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.orm import Session

import crud
import models
import schemas
from api import deps, tools

router = APIRouter()
cameras_logger = logger.bind(name="cameras")


@router.get("/get_cameras", response_model=List[schemas.CameraSelect])
async def get_faces(db: Session = Depends(deps.get_db)) -> Any:
    query = db.query(models.Camera).order_by(models.Camera.id).all()
    return query


class formAddCamera(BaseModel):
    cam_name: str
    cam_type: str
    cam_url: str
    cam_status: bool


@router.post("/add_camera")
async def add_camera(post_data: formAddCamera, db: Session = Depends(deps.get_db)) -> Any:
    if post_data.cam_type.lower() != post_data.cam_url.split(":")[0].lower():
        cameras_logger.error("The URL and type of the stream are inconsistent")
        return JSONResponse(status_code=423, content={"error": "The URL and type of the stream are inconsistent"})

    post_data = post_data.dict()
    cam_url = post_data["cam_url"]
    you = db.query(models.Camera).filter(models.Camera.cam_url == cam_url).first()
    if you:
        cameras_logger.warning(f"AddCamera | {cam_url} | warning: Cam-url already exists")
        return JSONResponse(status_code=409, content={"message": "Cam-url already exists"})
    else:
        cam_token = str(uuid.uuid4())
        cam_type = post_data["cam_type"]
        cam_status = post_data["cam_status"]
        if cam_type == 'rtsp':
            y, m = tools.check_rtsp_rtmp_stream(cam_url, is_rtmp=False)
            if y:
                post_data.update({'cam_status': True})
                cameras_logger.success(f"url:{cam_url} add success")
            else:
                post_data.update({'cam_status': False})
                cameras_logger.error(f"url:{cam_url} add filed | error:{m}")
        elif cam_type == 'rtmp':
            y, m = tools.check_rtsp_rtmp_stream(cam_url, is_rtmp=True)
            if y:
                post_data.update({'cam_status': True})
                cameras_logger.success(f"url:{cam_url} add success")
            else:
                post_data.update({'cam_status': False})
                cameras_logger.error(f"url:{cam_url} add filed | error:{m}")
        elif cam_type == 'webrtc':
            cameras_logger.info(f"webrtc url:{cam_url} added | {cam_status}")
        post_data.update({"cam_token": cam_token})
        camera = crud.crud_camera.create_camera(db=db, obj_in=post_data)
        return JSONResponse(status_code=200, content={"message": "Camera added"})


class formEditCamera(BaseModel):
    cam_name: str
    cam_type: str
    cam_url: str


@router.put("/update_camera/{cam_id}")
async def update_camera(cam_id: int, update_data: formEditCamera, db: Session = Depends(deps.get_db)) -> Any:
    if update_data.cam_type.lower() != update_data.cam_url.split(":")[0].lower():
        cameras_logger.error(f"UpdateCamera | The URL and type of the stream are inconsistent")
        return JSONResponse(status_code=423, content={"error": "The URL and type of the stream are inconsistent"})

    you = db.query(models.Camera).filter(models.Camera.cam_url == update_data.cam_url, models.Camera.id != cam_id).first()
    cam_name = update_data.cam_name
    cam_type = update_data.cam_type
    cam_url = update_data.cam_url
    if you:
        cameras_logger.warning(f"UpdateCamera | {update_data.cam_url} is already exists")
        return JSONResponse(status_code=409, content={"error": "The URL is already exists"})
    else:
        a = db.query(models.Camera).filter(models.Camera.id == cam_id).first()
        if cam_type == 'rtsp':
            y, m = tools.check_rtsp_rtmp_stream(cam_url, is_rtmp=False)
            cam_status = True if y else False
        elif cam_type == 'rtmp':
            y, m = tools.check_rtsp_rtmp_stream(cam_url, is_rtmp=True)
            cam_status = True if y else False
        else:
            cam_status = False
        a.cam_name = cam_name
        a.cam_url = cam_url
        a.cam_type = cam_type
        a.cam_status = cam_status
        db.commit()
        cameras_logger.success(f"UpdateCamera | {cam_url} is updated")
        return JSONResponse(status_code=200, content={"message": "Camera updated"})


@router.delete("/delete_camera/{camera_id}")
async def delete_camera(camera_id: int, db: Session = Depends(deps.get_db)):
    crud.crud_camera.delete_camera(db=db, id=camera_id)
    return JSONResponse(status_code=200, content={"message": "Camera deleted"})


@router.post("/checkRtspOrRtmp")
async def check_rtsp_or_rtmp(id: int = Form(...), cam_url: str = Form(...), cam_status: bool = Form(...),
                             cam_type: str = Form(...),
                             db: Session = Depends(deps.get_db)):
    is_rtmp = None
    if cam_type == 'rtsp':
        is_rtmp = False
    elif cam_type == 'rtmp':
        is_rtmp = True

    y, m = tools.check_rtsp_rtmp_stream(url=cam_url, is_rtmp=is_rtmp)
    if y:
        crud.crud_camera.update_camera_status(db=db, id=id, cam_status=y)
        return JSONResponse(status_code=200, content={'message': 'Camera is ready'})
    else:
        crud.crud_camera.update_camera_status(db=db, id=id, cam_status=y)
        return JSONResponse(status_code=404, content={'error': 'Camera is not functioning properly'})


@router.get("/checkAllRtspOrRtmp")
async def check_all_rtsp_or_rtmp(db: Session = Depends(deps.get_db)):
    all_cameras = db.query(models.Camera).all()

    for camera in all_cameras:
        print(camera.cam_url, camera.cam_type)
        cam_url = camera.cam_url
        cam_type = camera.cam_type
        if cam_type == 'rtsp' or cam_type == 'rtmp':
            is_rtmp = True if cam_type == 'rtmp' else False

            is_ready, message = tools.check_rtsp_rtmp_stream(url=cam_url, is_rtmp=is_rtmp)

            # 更新数据库中的状态
            camera.cam_status = is_ready
            db.commit()

    return JSONResponse(status_code=200, content={"message": "All cameras checked and updated"})
