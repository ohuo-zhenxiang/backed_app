import os
import schemas
import models
import crud
from api import deps
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.responses import JSONResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session
# from fastapi_pagination.ext.sqlalchemy import paginate
from fastapi_pagination import Page, paginate
from db.session import SessionLocal

router = APIRouter()


@router.get("/get_records/{token}", response_model=Page[schemas.RecordSelect])
def get_records(token: str, db: Session = Depends(deps.get_db), ):
    """
    Get all records
    """
    query = db.query(models.Record).where(models.Record.task_token == token).order_by(desc(models.Record.id)).all()
    return paginate(query)
