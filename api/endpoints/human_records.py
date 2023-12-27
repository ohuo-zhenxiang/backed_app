from fastapi import APIRouter, Depends
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate
from sqlalchemy import desc
from sqlalchemy.orm import Session

import models
import schemas
from api import deps

router = APIRouter()


@router.get("/get_human_records/{token}", response_model=Page[schemas.HumanRecordSelect])
def get_human_records(token: str, db: Session = Depends(deps.get_db)):
    """
    Get all human records
    """
    query = db.query(models.HumanRecord).where(models.HumanRecord.task_token == token).order_by(
        desc(models.HumanRecord.id))
    return paginate(query)
