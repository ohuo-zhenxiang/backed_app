from typing import Generator
from pydantic import ValidationError

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from sqlalchemy.orm import Session

import models, crud
from core.security import SECRET_KEY, ALGORITHM
from schemas import TokenPayload
from db.session import SessionLocal, async_session

reusable_oauth2 = OAuth2PasswordBearer(tokenUrl="api/login/access-token")


def get_db() -> Generator:
    try:
        with SessionLocal() as db:
            yield db
    finally:
        db.close()


async def get_async_db() -> Generator:
    try:
        async with async_session() as db:
            yield db
    finally:
        await db.close()


def get_current_user(db: Session = Depends(get_db), token: str = Depends(reusable_oauth2)) -> models.user.User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        token_data = TokenPayload(**payload)
    except (jwt.JWTError, ValidationError):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate credentials")
    user = crud.crud_user.get(db, id=token_data.sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def get_current_active_superuser(current_user: models.user.User = Depends(get_current_user)) -> models.user.User:
    if not crud.crud_user.is_superuser(current_user):
        raise HTTPException(status_code=400, detail="The user doesn't have enough privileges")
    return current_user
