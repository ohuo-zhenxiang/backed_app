import base64
import json
import os
from typing import List

import cv2
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
import schemas
import crud
import models
from api import deps
from settings import REDIS_URL, BASE_DIR
from loguru import logger
from datetime import datetime
import time
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

router = APIRouter()
overview_logger = logger.bind(name="overview")


@router.get("/get_cardData")
async def get_cardData(db: Session = Depends(deps.get_db)):
    faceWarehouse, faceGroup, task, equipment = {}, {}, {}, {}

    faceWarehouse["total"] = db.query(models.Face).count()
    faceWarehouse["uploadNum"] = db.query(func.count(models.Face.id)).filter(models.Face.source == "Upload").scalar()
    faceWarehouse["snapshotNum"] = db.query(func.count(models.Face.id)).filter(
        models.Face.source == "Snapshot").scalar()

    faceGroup["total"] = db.query(models.Group).count()
    faceGroup["used"] = (
        db.query(func.count(models.Group.id))
        .filter(models.Group.id == models.Task.associated_group_id)
        .group_by(models.Group.id)
        .count()
    )

    task["total"] = db.query(models.Task).count()
    task["running"] = db.query(func.count(models.Task.id)).filter(models.Task.status == "Running").scalar()
    task['finished'] = db.query(func.count(models.Task.id)).filter(models.Task.status == "Finished").scalar()
    task["waiting"] = db.query(func.count(models.Task.id)).filter(models.Task.status == "Waiting").scalar()

    equipment["total"] = 2
    equipment["normal"] = 1
    equipment["fault"] = 1

    cardData = {
        "faceWarehouse": faceWarehouse,
        "faceGroup": faceGroup,
        "task": task,
        "equipment": equipment}

    overview_logger.info("get CardData success")
    return JSONResponse(status_code=200, content=cardData)


@router.get("/get_recordList_by_taskToken/{taskToken}", response_model=List[schemas.RecordSelect])
def get_recordList_by_taskToken(taskToken: str, db: Session = Depends(deps.get_db)):
    """
    通过任务token获取任务记录列表，不带分页器的接口
    """
    records = db.query(models.Record).filter(models.Record.task_token == taskToken).order_by(
        desc(models.Record.id)).all()
    if records:
        overview_logger.success("get recordList success")
        return records
    else:
        overview_logger.error("get recordList failed")
        return JSONResponse(status_code=404, content="Not Found")


@router.get("/get_recordDrawImage/{recordId}")
def get_recordDrawImage(recordId: int, db: Session = Depends(deps.get_db)):
    """
    通过记录id获取记录的绘制图片
    """
    record = db.query(models.Record).filter(models.Record.id == recordId).first()
    if record:
        face_count = record.face_count
        record_image_path = record.record_image_path
        img = cv2.imread(os.path.join(BASE_DIR, record_image_path))
        if int(face_count) != 0:
            record_info = json.loads(record.record_info)
            for face in record_info["faces"]:
                box = face["box"]
                label = face["label"]
                if label == "UNK":
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=color, thickness=2)
                cv2.putText(img, f"{label}", (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)

        _, img_buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(img_buffer).decode("utf-8")
        return JSONResponse(status_code=200, content={"image_data": img_base64})
