from typing import Optional
from pydantic import BaseModel, validator


class UserBase(BaseModel):
    phone: Optional[str] = None

    # @validator('phone_number')
    # def validate_phone_number(cls, v):
    #     if not v.isdigit():
    #         raise ValueError('Phone number must contain only digits')
    #     if len(v) != 11:
    #         raise ValueError('Phone number must be 11 digits long')
    #     if not v.startswith('1'):
    #         raise ValueError('Phone number must start with 1')
    #     return v

    is_active: Optional[bool] = True
    is_superuser: bool = False
    full_name: Optional[str] = None


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
