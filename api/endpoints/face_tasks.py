import os
import shutil
import time
import uuid
from datetime import datetime, timedelta

from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from loguru import logger
from sqlalchemy import desc, func
from sqlalchemy.orm import Session, aliased
from typing import List
from pydantic import BaseModel

import crud
import models
import schemas
from api import deps, tools
from api.face_core.Feature_retrieval import AnnoyTree
from api.face_core.MainTask import SnapAnalysis, UpdateStatus
from db.session import SessionLocal
from scheduler_utils import Scheduler
from settings import TASK_RECORD_DIR, FEATURE_LIB, LOGGING_DIR
from redis_module import RedisModule

router = APIRouter()
tasks_logger = logger.bind(name="FaceTasks")


@router.get("/get_tasks", response_model=List[schemas.TaskSelect])
def get_tasks(db: Session = Depends(deps.get_db)):
    """
    Get all tasks .
    :param db:
    :return:
    """
    group_alias = aliased(models.Group)
    query = db.query(models.Task).order_by(desc(models.Task.id))
    query = query.outerjoin(group_alias, group_alias.id == models.Task.associated_group_id)
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
        models.Task.ex_detect,
        func.coalesce(group_alias.name, "None").label("associated_group_name"),
    ).all()
    tasks_logger.success("GetAllTasks successfully.")
    return query


@router.get("/get_task/{task_token}")
def get_task(task_token: str, db: Session = Depends(deps.get_db)):
    """
    Get task by task_token.
    :param db:
    :param task_token:
    :return:
    """
    query = db.query(models.Task).filter_by(task_token=task_token)
    tasks_logger.success(f"GetTaskByTaskToken successfully. task_token:{task_token}")
    return query.first()


def get_faces_feature_for_group(group_id: int):
    session = SessionLocal()
    group = session.query(models.Group).filter_by(id=group_id).first()
    id_name_features = [(face.id, face.name, face.face_features) for face in group.members]
    session.close()
    return id_name_features


class AddFaceTask(BaseModel):
    task_name: str
    ex_detect: list[str] = []
    interval_seconds: int
    start_time: int
    end_time: int
    capture_path: str
    associated_group_id: int


@router.post("/add_task")
def add_task(post_data:AddFaceTask, db: Session = Depends(deps.get_db)):
    """
    Add a new task to the scheduler.
    """
    task_name = post_data.task_name
    ex_detect = post_data.ex_detect
    interval_seconds = post_data.interval_seconds
    start_time = post_data.start_time
    end_time = post_data.end_time
    capture_path = post_data.capture_path
    associated_group_id = post_data.associated_group_id

    you = crud.crud_task.get_by_task_name(db, task_name=task_name)
    if you:
        tasks_logger.error(f"Task {task_name} already exists.")
        return JSONResponse(status_code=409, content={"message": "Task already exists."})
    else:
        stream_header = capture_path[:4]
        if stream_header != "rtsp" and stream_header != "rtmp":
            tasks_logger.error(f"{capture_path} is not a valid RTSP or RTMP URL.")
            return JSONResponse(status_code=410, content={"message": "capture_path is not a valid RTSP or RTMP URL."})
        is_rtmp = True if stream_header == "rtmp" else False
        s, m = tools.check_rtsp_rtmp_stream(url=capture_path, is_rtmp=is_rtmp)
        if not s:
            tasks_logger.error(f"{m}")
            return JSONResponse(status_code=410, content={"message": m})

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
                trigger = IntervalTrigger(seconds=interval_seconds, end_date=end_timestamp, timezone='Asia/Shanghai')
                crud.crud_task.create_task(db, task_name=task_name,
                                           ex_detect=ex_detect,
                                           task_token=task_token,
                                           interval_seconds=interval_seconds,
                                           start_time=start_time, end_time=end_time, capture_path=capture_path,
                                           status=status, associated_group_id=associated_group_id)
            else:
                status = "Waiting"
                trigger = IntervalTrigger(seconds=interval_seconds, start_date=start_timestamp, end_date=end_timestamp,
                                          timezone='Asia/Shanghai')
                crud.crud_task.create_task(db, task_name=task_name,
                                           ex_detect=ex_detect,
                                           task_token=task_token,
                                           interval_seconds=interval_seconds,
                                           start_time=start_time, end_time=end_time, capture_path=capture_path,
                                           status=status, associated_group_id=associated_group_id)
                Scheduler.add_job(UpdateStatus, trigger=DateTrigger(run_date=start_timestamp - timedelta(seconds=2),
                                                                    timezone='Asia/Shanghai'),
                                  args=[task_token, "Running"], id="START" + task_token, name="START" + task_name,
                                  executor="process",
                                  # jobstore='redis'
                                  )

            # 正餐
            job_create = Scheduler.add_job(
                SnapAnalysis, trigger,
                args=[task_token, ex_detect, capture_path, save_fold],
                id=task_token, name=task_name, executor="process",
                # jobstore='redis'
            )

            Scheduler.add_job(UpdateStatus,
                              trigger=DateTrigger(run_date=end_timestamp + timedelta(seconds=2), timezone='Asia/Shanghai'),
                              args=[task_token, "Finished"], id="END" + task_token, name="END" + task_name,
                              executor="process",
                              # jobstore='redis'
                              )

            tasks_logger.success(f"Task {task_token} created")
            return JSONResponse(status_code=200, content={"task_token": task_token, "detail": 'Job created'})
        else:
            tasks_logger.warning(f"id-{b}| name-{c}: size is wrong")
            return JSONResponse(content={"error_message": f"id-{b}| name-{c}: size is wrong"}, status_code=400)


@router.delete("/delete_task/{task_token}")
def delete_task(task_token: str, background_tasks: BackgroundTasks, db: Session = Depends(deps.get_db)):
    """
    删除任务
    """
    background_tasks.add_task(delete_task_async, task_token)
    crud.crud_task.delete_task(db, task_token)
    tasks_logger.success(f"DeletedTask {task_token}")
    return JSONResponse(status_code=200, content={"detail": 'Task deleted'})


def safe_remove_file(file_path):
    max_attempts = 5
    attempts = 0
    while attempts < max_attempts:
        try:
            os.remove(file_path)
            return True
        except OSError:
            attempts += 1
            time.sleep(1)
    return False


async def delete_task_async(task_token: str):
    """
    删除任务
    """

    # 1. stop and remove scheduler_job
    if Scheduler.get_job(task_token):
        Scheduler.remove_job(task_token)
        time.sleep(1)

    # 2. remove .ann file and pickle file (redis)
    try:
        shutil.rmtree(os.path.join(TASK_RECORD_DIR, task_token))
    except (PermissionError, FileNotFoundError) as e:
        print(e)
    ann_deleted = safe_remove_file(os.path.join(FEATURE_LIB, f"{task_token}.ann"))

    with RedisModule() as rds:
        res_bool = rds.delete(f"Pickle_{task_token}")

    if ann_deleted and res_bool:
        tasks_logger.success(f"Task file deleted successfully")
    else:
        tasks_logger.error(f"Task file delete failed")
