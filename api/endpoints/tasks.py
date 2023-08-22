from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.responses import JSONResponse

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import time
import uuid
import json
from main import Scheduler


router = APIRouter()


@router.post("/create_task")
def add_task(task_name: str, interval_seconds: int, start_time):
    """
    Add a new task to the scheduler.
    """


