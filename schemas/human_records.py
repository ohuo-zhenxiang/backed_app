from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel


class HumanRecordBase(BaseModel):
    id: int
    start_time: datetime
    human_count: int
    record_image_path: Optional[str]


class HumanInfo(HumanRecordBase):
    box: List[int]
    detect_score: float
    label_id: int


class HumanRecordInfo(HumanRecordBase):
    humans: List[HumanInfo]
    task_status: str
    humans_count: int


class HumanRecordSelect(HumanRecordBase):
    record_info: str

    class Config:
        from_attributes = True


class HumanRecordRead(HumanRecordBase):
    pass


class HumanRecordCreate(HumanRecordBase):
    pass


class HumanRecordUpdate(HumanRecordBase):
    pass
