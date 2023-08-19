from typing import Optional
from pydantic import BaseModel
from datetime import datetime


class GroupBase(BaseModel):
    name: str
    description: str = None


class GroupSelect(GroupBase):
    id: int

    class Config:
        orm_mode = True


class GroupCreate(GroupBase):
    pass


class GroupUpdate(GroupBase):
    pass



