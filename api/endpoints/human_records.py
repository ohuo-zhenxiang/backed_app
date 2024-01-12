import base64
import json
import cv2
import numpy as np
import os
from fastapi import APIRouter, Depends
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate
from sqlalchemy import desc
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse

import models
import schemas
from api import deps
from api.human_core.Pose_detect import vis_frame
from settings import BASE_DIR

router = APIRouter()


@router.get("/get_human_records/{token}", response_model=Page[schemas.HumanRecordSelect])
def get_human_records(token: str, db: Session = Depends(deps.get_db)):
    """
    Get all human records
    """
    query = db.query(models.HumanRecord).where(models.HumanRecord.task_token == token).order_by(
        desc(models.HumanRecord.id))
    return paginate(query)


@router.get("/get_recordPoseImage/{record_id}")
def get_recordPoseImage(record_id: int, db: Session = Depends(deps.get_db)):
    """
    通过记录id先绘制姿态的图片
    """
    record = db.query(models.HumanRecord).filter(models.HumanRecord.id == record_id).first()
    if record:
        human_count = record.human_count
        record_image_path = record.record_image_path
        raw_img = cv2.imread(os.path.join(BASE_DIR, record_image_path))
        record_info = json.loads(record.record_info)
        if int(human_count) != 0:
            humans_info = record_info['humans']
            poses_coords, poses_scores = [], []
            for person in humans_info:
                person_poses = person['person_poses']
                pose_coords, pose_scores = person_poses["pose_coords"], person_poses["pose_scores"]
                poses_coords.append(np.array(pose_coords))
                poses_scores.append(np.array(pose_scores).reshape(-1, 1))
            raw_img = vis_frame(raw_img, poses_coords, poses_scores)

    _, img_buffer = cv2.imencode('.jpg', raw_img)
    img_base64 = base64.b64encode(img_buffer).decode("utf-8")
    return JSONResponse(status_code=200, content={"image_data": img_base64})
