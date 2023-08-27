import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.responses import JSONResponse
from sqlalchemy import desc

import schemas
import crud
import models
from api import deps
from sqlalchemy.orm import Session
from scheduler_utils import Scheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi_pagination.ext.sqlalchemy import paginate
from fastapi_pagination import Page
from api.face_core.MainTask import SnapAnalysis
from settings import TASK_RECORD_DIR, FEATURE_LIB
from db.session import SessionLocal
from api.face_core.Feature_retrieval import AnnoyTree

from datetime import datetime
import time
import uuid
import json

router = APIRouter()


@router.get("/get_tasks", response_model=Page[schemas.TaskSelect])
def get_tasks(db: Session = Depends(deps.get_db)):
    """
    Get all tasks .
    :param db:
    :return:
    """
    query = db.query(models.Task).order_by(desc(models.Task.id))
    return paginate(query)


def get_faces_feature_for_group(group_id: int):
    session = SessionLocal()
    group = session.query(models.Group).filter_by(id=group_id).first()
    id_name_features = [(face.id, face.name, face.face_features) for face in group.members]
    session.close()
    return id_name_features


@router.post("/add_task")
def add_task(task_name: str = Form(...), interval_seconds: int = Form(...),
             start_time: int = Form(...), end_time: int = Form(...),
             capture_path: str = Form(...), associated_group_id: int = Form(...),
             db: Session = Depends(deps.get_db)):
    """
    Add a new task to the scheduler.
    """
    task_token = str(uuid.uuid4())
    capture_path = capture_path.replace("\\", "/")
    start_time = datetime.fromtimestamp(start_time/1000).strftime("%Y-%m-%d %H:%M:%S")
    end_time = datetime.fromtimestamp(end_time/1000).strftime("%Y-%m-%d %H:%M:%S")

    id_name_features = get_faces_feature_for_group(associated_group_id)
    ann_tree = AnnoyTree()
    a, b, c = ann_tree.create_tree(vector_list=id_name_features, save_filename=task_token)
    if a:
        # 创建定时间隔任务
        save_fold = os.path.join(TASK_RECORD_DIR, task_token)
        os.makedirs(save_fold)
        crud.crud_task.create_task(db, task_name=task_name, task_token=task_token, interval_seconds=interval_seconds,
                                   start_time=start_time, end_time=end_time, capture_path=capture_path)
        trigger = IntervalTrigger(seconds=interval_seconds, start_date=start_time, end_date=end_time)
        job_create = Scheduler.add_job(
            SnapAnalysis, trigger,
            args=[task_token, capture_path, save_fold],
            id=task_token, name=task_name, executor="process",
            # jobstore='redis'
        )
        print(start_time, end_time)
        return JSONResponse(status_code=200, content={"task_token": task_token, "detail": 'Job created'})
    else:
        return JSONResponse(content={"error_message": f"id-{b}| name-{c}: size is wrong"}, status_code=400)


@router.delete("/delete_task/{task_token}")
def delete_task(task_token: str, db: Session = Depends(deps.get_db)):
    """
    删除任务
    """
    crud.crud_task.delete_task(db, task_token)
    shutil.rmtree(os.path.join(TASK_RECORD_DIR, task_token))
    os.remove(os.path.join(FEATURE_LIB, f"{task_token}.ann"))
    os.remove(os.path.join(FEATURE_LIB, f"{task_token}.pickle"))

    return JSONResponse(status_code=200, content={"detail": 'Task deleted'})