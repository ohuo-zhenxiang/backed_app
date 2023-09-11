from typing import Any
from loguru import logger
from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session

import schemas, crud, core, models
from api import deps

router = APIRouter()
users_logger = logger.bind(name="Users")


@router.post("/create_user", response_model=schemas.User)
def create_user(*, db: Session = Depends(deps.get_db), user_in: schemas.UserCreate,
                current_user: models.user.User = Depends(deps.get_current_active_superuser)) -> Any:
    user = crud.crud_user.get_by_phone(db, phone=user_in.phone)
    if user:
        raise HTTPException(status_code=400, detail="The user with this phone already exists in the system.")
    user = crud.crud_user.create(db, obj_in=user_in)
    users_logger.success(f"User {user_in.phone} created successfully.")
