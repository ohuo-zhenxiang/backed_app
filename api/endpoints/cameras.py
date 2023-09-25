from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc
from loguru import logger
import numpy as np
import uuid
import os
import time
import base64

import schemas, crud, models
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
        elif cam_type == 'rtmp':
            y, m = tools.chechk_rtsp_rtmp_stream(cam_url, is_rtmp=True)
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


@router.put("/update_camera/{camera_id}")
async def update_camera():
    pass


@router.delete("/delete_camera/{camera_id}")
async def delete_camera(camera_id: int, db: Session = Depends(deps.get_db)):
    crud.crud_camera.delete_camera(db=db, id=camera_id)
    return JSONResponse(status_code=200, content={"message": "Camera deleted"})


@router.post("/checkRtspOrRtmp")
async def check_rtsp_or_rtmp(id: int = Form(...), cam_url: str = Form(...), cam_status: bool = Form(...),
                             cam_type: str = Form(...),
                             db: Session = Depends(deps.get_db)):
    is_rtmp = None
    s = time.time()
    if cam_type == 'rtsp':
        is_rtmp = False
    elif cam_type == 'rtmp':
        is_rtmp = True

    y, m = tools.check_rtsp_rtmp_stream(url=cam_url, is_rtmp=is_rtmp)
    if y:
        if y != cam_status:
            crud.crud_camera.update_camera_status(db=db, id=id, cam_status=y)
        return JSONResponse(status_code=200, content={'message': 'Camera is ready'})
    else:
        if y != cam_status:
            crud.crud_camera.update_camera_status(db, id, y)
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
