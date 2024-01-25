import os
import shutil
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Union

from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import APIRouter, Depends, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import desc
from sqlalchemy.orm import Session
from pydantic import BaseModel

import crud
import models
import schemas
from api import deps
from api.human_core.MultiMainTask import SnapMultiAnalysis, MultiUpdateStatus
from scheduler_utils import Scheduler
from settings import TASK_RECORD_DIR

router = APIRouter()
human_tasks_logger = logger.bind(name="HumanTasks")


@router.get("/get_human_tasks", response_model=List[schemas.HumanTaskSelect])
async def get_human_tasks(db: AsyncSession = Depends(deps.get_db)):
    """
    Get all human-tasks.
    """
    query = db.query(models.HumanTask).order_by(desc(models.HumanTask.id)).all()
    return query


@router.get("/get_HumanTask_ByToken/{task_token}")
async def get_human_task_by_token(task_token: str, db: AsyncSession = Depends(deps.get_db)):
    """
    Get human-task by task_token
    :param task_token:
    :param db:
    :return:
    """
    query = db.query(models.HumanTask).filter_by(task_token=task_token)
    human_tasks_logger.success(f"GetHumanTaskByTaskToken successfully, task_token: {task_token}")
    return query.first()


class AddHumanTask(BaseModel):
    task_name: str
    task_extends: List[str] = []
    interval_seconds: int
    start_time: int
    end_time: int
    capture_path: str


@router.post("/add_human_task")
def add_human_task(data: AddHumanTask, db: Session = Depends(deps.get_db)):
    """
    Add a new task to the scheduler.
    """
    task_name = data.task_name
    task_extends = data.task_extends[1:]
    interval_seconds = data.interval_seconds
    start_time = data.start_time
    end_time = data.end_time
    capture_path = data.capture_path

    you = crud.crud_human_task.get_HumanTask_by_TaskName(db, task_name)
    if you:
        human_tasks_logger.error(f"HumanTask: {task_name} already exists")
        return JSONResponse(status_code=409, content={"message": "HumanTask already exists"})
    else:
        task_token = str(uuid.uuid4())

        capture_path = capture_path.replace("\\", "/")
        start_timestamp = datetime.fromtimestamp(start_time / 1000)
        start_time = start_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        end_timestamp = datetime.fromtimestamp(end_time / 1000)
        end_time = end_timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # 创建定时间隔任务
        save_fold = os.path.join(TASK_RECORD_DIR, task_token)
        os.makedirs(save_fold)

        if start_timestamp < datetime.now():
            status = "Running"
            trigger = IntervalTrigger(seconds=interval_seconds, end_date=end_timestamp, timezone='Asia/Shanghai')
            crud.crud_human_task.create_human_task(db, task_name=task_name,
                                                   expand_tasks=task_extends,
                                                   task_token=task_token,
                                                   interval_seconds=interval_seconds, start_time=start_time,
                                                   end_time=end_time, capture_path=capture_path, status=status)
        else:
            status = "Waiting"
            trigger = IntervalTrigger(seconds=interval_seconds, start_date=start_timestamp,
                                      end_date=end_timestamp, timezone='Asia/Shanghai')
            crud.crud_human_task.create_human_task(db, task_name=task_name,
                                                   expand_tasks=task_extends,
                                                   task_token=task_token,
                                                   interval_seconds=interval_seconds, start_time=start_time,
                                                   end_time=end_time, capture_path=capture_path, status=status)
            Scheduler.add_job(MultiUpdateStatus,
                              trigger=DateTrigger(
                                  run_date=start_timestamp - timedelta(seconds=2),
                                  timezone='Asia/Shanghai'),
                              args=[task_token, "Running"], id=f"START-{task_token}", name=f"START-{task_name}",
                              executor="process",
                              # jobstore='redis'
                              )

        # 正餐

        job_create = Scheduler.add_job(
            SnapMultiAnalysis, trigger,
            args=[task_token, task_extends, capture_path, save_fold],
            id=task_token, name=task_name, executor="process",
            # jobstore='redis'
        )

        Scheduler.add_job(MultiUpdateStatus,
                          trigger=DateTrigger(run_date=end_timestamp + timedelta(seconds=2), timezone='Asia/Shanghai'),
                          args=[task_token, "Finished"], id=f"END-{task_token}", name=f"END-{task_name}",
                          executor="process",
                          # jobstore='redis'
                          )
        human_tasks_logger.success(f"Task {task_token} created")
        return JSONResponse(status_code=200, content={"task_token": task_token, "detail": "HumanTask created"})


@router.delete("/delete_human_task/{task_token}")
async def delete_human_task(task_token: str, background_tasks: BackgroundTasks, db: Session = Depends(deps.get_db)):
    """
    删除任务
    """
    background_tasks.add_task(delete_human_task_async, task_token, db)
    human_tasks_logger.success(f"DeleteHumanTask: {task_token}")
    return JSONResponse(status_code=200, content={"detail": "HumanTask deleted"})


async def delete_human_task_async(task_token: str, db: Session):
    """
    删除任务记录文件夹
    """
    if Scheduler.get_job(task_token):
        Scheduler.remove_job(task_token)
        time.sleep(1)
    crud.crud_human_task.delete_human_task_by_token(db, task_token)
    shutil.rmtree(os.path.join(TASK_RECORD_DIR, task_token))
    human_tasks_logger.success(f"DeleteHumanTask: {task_token}")


if __name__ == '__main__':
    pass
