import os

from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.responses import JSONResponse

import schemas
import crud
from api import deps
from sqlalchemy.orm import Session
from scheduler_utils import Scheduler
from apscheduler.triggers.interval import IntervalTrigger
from api.face_core.MainTask import SnapAnalysis
from settings import TASK_RECORD_DIR

from datetime import datetime
import time
import uuid
import json

router = APIRouter()

detector_path = '../face_core/model/det_10g.onnx'
recognizer_path = '../face_core/model/w600k_r50.onnx'


@router.post("/create_task")
def add_task(task_name: str, interval_seconds: int, start_time: str, end_time: str, capture_path: str,
             db: Session = Depends(deps.get_db)):
    """
    Add a new task to the scheduler.
    """
    task_token = str(uuid.uuid4())
    capture_path = capture_path.replace("\\", "/")
    save_fold = os.path.join(TASK_RECORD_DIR, task_token)
    os.makedirs(save_fold)

    crud.crud_task.create_task(db, task_name=task_name, task_token=task_token, interval_seconds=interval_seconds,
                               start_time=start_time, end_time=end_time, capture_path=capture_path)
    trigger = IntervalTrigger(seconds=interval_seconds, start_date=start_time, end_date=end_time)
    job_create = Scheduler.add_job(SnapAnalysis, trigger,
                                   args=[task_token, capture_path, save_fold],
                                   id=task_token, name=task_name, executor="process")
    return JSONResponse(status_code=200, content={"task_token": task_token})
