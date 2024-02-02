from typing import Optional
from pydantic import BaseModel, validator


class UserBase(BaseModel):
    phone: Optional[str] = None
    is_active: Optional[bool] = True
    is_superuser: bool = False
    full_name: Optional[str] = None
    permissions: Optional[str] = None


class UserCreate(UserBase):
    phone: str
    password: str


class UserUpdate(UserBase):
    id: Optional[int] = None

    class Config:
        from_attributes = True


class UserInDBBase(UserBase):
    id: Optional[int] = None

    class Config:
        from_attributes = True


class User(UserInDBBase):
    pass


class UserInDB(UserInDBBase):
    hashed_password: str
