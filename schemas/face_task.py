from typing import Optional, List, Any
from pydantic import BaseModel
from datetime import datetime


class TaskBase(BaseModel):
    task_token: str
    name: str
    start_time: datetime
    end_time: datetime
    interval_seconds: int
    status: str
    associated_group_id: int
    capture_path: Optional[str]


class TaskCreate(TaskBase):
    """
    This is the request body for creating a task
    """
    pass


class TaskSelect(TaskBase):
    id: Optional[int] = None
    created_time: datetime
    associated_group_name: str

    class Config:
        from_attributes = True


class TaskUpdate(TaskBase):
    """
    This is the request body for updating a task
    """
    pass
