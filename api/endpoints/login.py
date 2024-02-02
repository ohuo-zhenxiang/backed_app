from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from datetime import timedelta
from jose import jwt, JWTError
from typing import Any

import schemas, models, crud
from core import security
from api import deps
from settings import ACCESS_TOKEN_EXPIRE_MINUTES
from core.security import ALGORITHM, SECRET_KEY
from loguru import logger

login_logger = logger.bind(name="Login")

router = APIRouter()


@router.post('/access-token', response_model=schemas.Token)
def login_access_token(db: Session = Depends(deps.get_db), form_data: OAuth2PasswordRequestForm = Depends()) -> Any:
    '''
    OAuth2 compatible token login, get an access token for future requests
    '''
    # Check if the user exists in the database
    user = crud.crud_user.authenticate(db, phone=form_data.username, password=form_data.password)
    # If the user doesn't exist, return an error
    if not user:
        return JSONResponse(status_code=401,  content={"detail": "Incorrect username or password"})
    # If the user is inactive, return an error
    elif not crud.crud_user.is_activate(user):
        return JSONResponse(status_code=401, content={"detail": "Inactive user"})
    # Create a token with the user's id and expiration time
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    login_logger.success(f"{form_data.username} login success")
    return {"access_token": security.create_access_token(user.id, expires_delta=access_token_expires),
            "token_type": "bearer"}


@router.get('/getUserInfo', response_model=schemas.user.User)
def get_user_info(authorization: str = Header(..., convert_underscores=False), db: Session = Depends(deps.get_db)):
    _, token = authorization.split(' ')
    try:
        user_info = deps.get_current_user(token, db)
    except Exception as e:
        login_logger.error(f"get user info error: {e}")
        return JSONResponse(status_code=401, content={"error": f"{e}"})
    else:
        return user_info

