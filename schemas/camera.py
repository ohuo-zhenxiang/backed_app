from typing import Optional
from pydantic import BaseModel
from datetime import datetime


class CameraBase(BaseModel):
    cam_name: Optional[str] = None
    cam_type: Optional[str] = None
    cam_url: Optional[str] = None
    cam_status: Optional[bool] = None


class CameraSelect(CameraBase):
    id: Optional[int]
    update_time: datetime
    created_time: datetime

    class Config:
        from_attributes = True


class CameraCreate(CameraBase):
    cam_token: Optional[str]


class CameraUpdate(CameraBase):
    pass


class CameraDelete(BaseModel):
    id: Optional[int] = None
