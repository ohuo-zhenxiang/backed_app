import os
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from sqlalchemy.orm import Session
from fastapi_pagination.ext.sqlalchemy import paginate, select
from fastapi_pagination import Page
from settings import UPLOAD_DIR

import schemas, crud, core, models
from api import deps

router = APIRouter()


@router.get("/get_faces", response_model=Page[schemas.FaceSelect])
async def get_faces(db: Session = Depends(deps.get_db)) -> Any:
    query = db.query(models.Face)
    return paginate(query)


@router.post("/add_face")
def create_face(*, db: Session = Depends(deps.get_db), face_in: schemas.FaceCreate) -> Any:
    face_in.face_features = b'aaaa'
    crud.crud_face.create(db, obj_in=face_in)
    return {"message": "Face added"}


@router.post("/add_test_face")
async def create_test_face(file: UploadFile = File(...), name: str = Form(...), phone: str = Form(...)) -> Any:
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    print(name, phone)
    print(type(name), type(phone), type(file))
    return {"message": "Face added"}
