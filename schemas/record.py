from typing import Optional
from pydantic import BaseModel
from datetime import datetime


class RecordBase(BaseModel):
    id: int
    start_time: datetime
    face_count: int
    record_image_path: Optional[str]


class RecordSelect(RecordBase):
    class Config:
        orm_mode = True


class RecordRead(RecordBase):
    pass


class RecordCreate(RecordBase):
    pass


class RecordUpdate(RecordBase):
    pass