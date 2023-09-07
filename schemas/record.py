from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime


class RecordBase(BaseModel):
    id: int
    start_time: datetime
    face_count: int
    record_image_path: Optional[str]


class FaceInfo(BaseModel):
    box: List[int]
    detect_score: float
    kps: List[List[float]]
    label: str
    label_id: int


class RecordInfo(BaseModel):
    faces: List[FaceInfo]
    task_status: str
    faces_count: int


class RecordSelect(RecordBase):
    record_names: List[str]
    record_info: str

    class Config:
        orm_mode = True


class RecordRead(RecordBase):
    pass


class RecordCreate(RecordBase):
    pass


class RecordUpdate(RecordBase):
    pass
