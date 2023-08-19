from typing import Optional
from pydantic import BaseModel
from datetime import datetime


class FaceBase(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    gender: Optional[str] = None
    face_image_path: Optional[str] = None
    face_features: Optional[bytes] = None


class FaceSelect(FaceBase):
    id: Optional[int]
    created_time: datetime

    class Config:
        orm_mode = True


class FaceCreate(FaceBase):
    pass


class FaceUpdate(FaceBase):
    pass


class FaceInDBBase(FaceBase):
    id: Optional[int] = None

    class Config:
        orm_mode = True


class MembersInGroup(BaseModel):
    value: int
    label: str

    class Config:
        orm_mode = True

# Additional properties to return via API
