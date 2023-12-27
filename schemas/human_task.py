from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class HumanTaskBase(BaseModel):
    task_token: str
    task_name: str
    start_time: datetime
    end_time: datetime
    interval_seconds: int
    status: str
    capture_path: Optional[str]


class HumanTaskCreate(HumanTaskBase):
    """
    This is the request body for creating a task
    """
    pass


class HumanTaskSelect(HumanTaskBase):
    id: Optional[int] = None
    created_time: datetime

    class Config:
        from_attributes = True


class HumanTaskUpdate(HumanTaskBase):
    """
    This is the request body for updating a task
    """
    pass
