import os
import shutil
import time
import uuid
from datetime import datetime, timedelta
from typing import List

from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import APIRouter, Depends, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate
from loguru import logger
from sqlalchemy import desc
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

import crud
import models
import schemas
from api import deps
from api.human_core.HumanMainTask import SnapHumanAnalysis, UpdateStatus
from scheduler_utils import Scheduler
from settings import TASK_RECORD_DIR

router = APIRouter()
human_tasks_logger = logger.bind(name="HumanTasks")


@router.get("/get_human_tasks", response_model=List[schemas.HumanTaskSelect])
async def get_human_tasks(db: AsyncSession = Depends(deps.get_async_db)):
    """
    Get all human-tasks.
    """
    query = await db.execute(models.HumanTask.__table__.select().order_by(models.HumanTask.id.desc()))
    human_tasks = query.fetchall()
    human_tasks_logger.success("GetAllHumanTasks successfully")
    return human_tasks


@router.get("/get_HumanTask_ByToken/{task_token}")
async def get_human_task_by_token(task_token: str, db: AsyncSession = Depends(deps.get_async_db)):
    """
    Get human-task by task_token
    :param task_token:
    :param db:
    :return:
    """
    stmt = select(models.HumanTask).filter_by(task_token=task_token)
    result = await db.execute(stmt)
    human_task = result.scalar_one_or_none()
    human_tasks_logger.success(f"GetHumanTaskByTaskToken successfully, task_token: {task_token}")
    return human_task


@router.post("/add_human_task")
def add_human_task(task_name: str = Form(...), interval_seconds: int = Form(...),
                   start_time: int = Form(...), end_time: int = Form(...),
                   capture_path: str = Form(...), db: Session = Depends(deps.get_db)):
    """
    Add a new task to the scheduler.
    """
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
            crud.crud_human_task.create_human_task(db, task_name=task_name, task_token=task_token,
                                                   interval_seconds=interval_seconds, start_time=start_time,
                                                   end_time=end_time, capture_path=capture_path, status=status)
        else:
            status = "Waiting"
            trigger = IntervalTrigger(seconds=interval_seconds, start_date=start_timestamp,
                                      end_date=end_timestamp, timezone='Asia/Shanghai')
            crud.crud_human_task.create_human_task(db, task_name=task_name, task_token=task_token,
                                                   interval_seconds=interval_seconds, start_time=start_time,
                                                   end_time=end_time, capture_path=capture_path, status=status)
            Scheduler.add_job(UpdateStatus, trigger=DateTrigger(run_date=start_timestamp - timedelta(seconds=2),
                                                                timezone='Asia/Shanghai'),
                              args=[task_token, "Running"], id=f"START-{task_token}", name=f"START-{task_name}",
                              executor="process",
                              # jobstore='redis'
                              )

        # 正餐
        job_create = Scheduler.add_job(
            SnapHumanAnalysis, trigger,
            args=[task_token, capture_path, save_fold],
            id=task_token, name=task_name, executor="process",
            # jobstore='redis'
        )

        Scheduler.add_job(UpdateStatus,
                          trigger=DateTrigger(run_date=end_timestamp + timedelta(seconds=2), timezone='Asia/Shanghai'),
                          args=[task_token, "Finished"], id=f"END-{task_token}", name=f"END-{task_name}",
                          executor="process",
                          # jobstore='redis'
                          )
        human_tasks_logger.success(f"Task {task_token} created")
        return JSONResponse(status_code=200, content={"task_token": task_token, "detail": "HumanTask created"})


@router.delete("/delete_human_task/{task_token}")
async def delete_human_task(task_token: str, background_tasks: BackgroundTasks, db: Session = Depends(deps.get_async_db)):
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
