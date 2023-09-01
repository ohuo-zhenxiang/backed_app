import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.responses import JSONResponse
from sqlalchemy import desc

import schemas
import crud
import models
from api import deps
from sqlalchemy.orm import Session, joinedload, aliased
from scheduler_utils import Scheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from fastapi_pagination.ext.sqlalchemy import paginate
from fastapi_pagination import Page
from api.face_core.MainTask import SnapAnalysis, UpdateStatus
from settings import TASK_RECORD_DIR, FEATURE_LIB
from db.session import SessionLocal
from api.face_core.Feature_retrieval import AnnoyTree

from datetime import datetime, timedelta
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
    group_alias = aliased(models.Group)
    query = db.query(models.Task).order_by(desc(models.Task.id))
    query = query.join(group_alias, group_alias.id == models.Task.associated_group_id)
    query = query.add_columns(
        models.Task.id,
        models.Task.task_token,
        models.Task.name,
        models.Task.status,
        models.Task.associated_group_id,
        models.Task.start_time,
        models.Task.end_time,
        models.Task.capture_path,
        models.Task.created_time,
        models.Task.interval_seconds,
        group_alias.name.label("associated_group_name"),

    )
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
    start_timestamp = datetime.fromtimestamp(start_time / 1000)
    start_time = start_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    end_timestamp = datetime.fromtimestamp(end_time / 1000)
    end_time = end_timestamp.strftime("%Y-%m-%d %H:%M:%S")

    id_name_features = get_faces_feature_for_group(associated_group_id)
    ann_tree = AnnoyTree()
    a, b, c = ann_tree.create_tree(vector_list=id_name_features, save_filename=task_token)
    if a:
        # 创建定时间隔任务
        save_fold = os.path.join(TASK_RECORD_DIR, task_token)
        os.makedirs(save_fold)

        if start_timestamp < datetime.now():
            status = "Running"
            trigger = IntervalTrigger(seconds=interval_seconds, end_date=end_timestamp)
            crud.crud_task.create_task(db, task_name=task_name, task_token=task_token,
                                       interval_seconds=interval_seconds,
                                       start_time=start_time, end_time=end_time, capture_path=capture_path,
                                       status=status, associated_group_id=associated_group_id)
        else:
            status = "Waiting"
            trigger = IntervalTrigger(seconds=interval_seconds, start_date=start_timestamp, end_date=end_timestamp)
            crud.crud_task.create_task(db, task_name=task_name, task_token=task_token,
                                       interval_seconds=interval_seconds,
                                       start_time=start_time, end_time=end_time, capture_path=capture_path,
                                       status=status, associated_group_id=associated_group_id)
            Scheduler.add_job(UpdateStatus, trigger=DateTrigger(run_date=start_timestamp + timedelta(seconds=2)),
                              args=[task_token, "Running"], id="START" + task_token, name="START" + task_name,
                              executor="process",
                              # jobstore='redis'
                              )

        job_create = Scheduler.add_job(
            SnapAnalysis, trigger,
            args=[task_token, capture_path, save_fold],
            id=task_token, name=task_name, executor="process",
            # jobstore='redis'
        )

        Scheduler.add_job(UpdateStatus, trigger=DateTrigger(run_date=end_timestamp + timedelta(seconds=2)),
                          args=[task_token, "Finished"], id="END" + task_token, name="END" + task_name,
                          executor="process",
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
    if Scheduler.get_job(task_token):
        Scheduler.remove_job(task_token)
        time.sleep(0.5)
    crud.crud_task.delete_task(db, task_token)
    shutil.rmtree(os.path.join(TASK_RECORD_DIR, task_token))
    os.remove(os.path.join(FEATURE_LIB, f"{task_token}.ann"))
    os.remove(os.path.join(FEATURE_LIB, f"{task_token}.pickle"))

    return JSONResponse(status_code=200, content={"detail": 'Task deleted'})
