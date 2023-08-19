import os
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.responses import JSONResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session
from fastapi_pagination.ext.sqlalchemy import paginate, select
from fastapi_pagination import Page
from settings import UPLOAD_DIR
import uuid

import schemas, crud, core, models
from api import deps

router = APIRouter()


@router.get("/get_faces", response_model=Page[schemas.FaceSelect])
async def get_faces(db: Session = Depends(deps.get_db)) -> Any:
    query = db.query(models.Face).order_by(desc(models.Face.id))
    return paginate(query)


# @router.post("/add_face_test")
# def create_face(*, db: Session = Depends(deps.get_db), face_in: schemas.FaceCreate) -> Any:
#     face_in.face_features = b'aaaa'
#     crud.crud_face.create(db, obj_in=face_in)
#     return {"message": "Face added"}


@router.post("/add_face")
async def create_face(file: UploadFile = File(...), name: str = Form(...), phone: str = Form(...),
                           db: Session = Depends(deps.get_db)) -> Any:
    you = crud.crud_face.get_by_phone(db, phone=phone)
    image_name = str(uuid.uuid1()) + '.jpg'

    if not you:
        file_save_path = os.path.join(UPLOAD_DIR, image_name)
        with open(file_save_path, "wb+") as f:
            f.write(await file.read())
        face_in = schemas.FaceCreate(name=name, phone=phone, face_features=b'aaaa', face_image_path=file_save_path)
        crud.crud_face.create_face(db, obj_in=face_in)
        return JSONResponse(status_code=200, content={"message": "Face added"})
    else:
        return JSONResponse(status_code=400, content={"message": "Phone already exists"})


@router.delete("/delete_face/{face_id}")
async def delete_face(face_id: int, db: Session = Depends(deps.get_db)) -> Any:
    a = crud.crud_face.delete_face_by_id(db, id=face_id)
    if a:
        return JSONResponse(status_code=200, content={"message": "Face deleted"})


@router.put("/update_face/{face_id}")
async def update_face(face_id: int, name: str = Form(...), phone: str = Form(...), db: Session = Depends(deps.get_db)) -> Any:
    # face = crud.crud_face.get(db, id=face_id)
    # print(face)
    # print(name, phone)
    a = crud.crud_face.update_face_by_id(db, face_id=face_id, name=name, phone=phone)
    if a:
        return JSONResponse(status_code=200, content={"message": "Face updated"})
    else:
        return JSONResponse(status_code=400, content={"message": "Phone already exists"})
