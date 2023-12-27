from typing import Optional
from pydantic import BaseModel
from datetime import datetime


class GroupBase(BaseModel):
    name: str
    description: str = None


class GroupSelect(GroupBase):
    id: int
    member_count: Optional[int]

    class Config:
        from_attributes = True


class GroupCreate(GroupBase):
    pass


class GroupUpdate(GroupBase):
    pass



