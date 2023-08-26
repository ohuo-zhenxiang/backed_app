from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.responses import JSONResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session
from fastapi_pagination.ext.sqlalchemy import paginate, select
from fastapi_pagination import Page
from settings import UPLOAD_DIR
from api.face_core.RetinaFace_detect import RetinaFace
from api.face_core.ArcFace_extract import ArcFaceOrt
from logger_module import Logger
import uuid
import cv2
import os
import pickle
import numpy as np
import time
import datetime

import schemas, crud, core, models
from api import deps

router = APIRouter()
Detector = RetinaFace()
Recognizer = ArcFaceOrt()


@router.get("/get_faces", response_model=Page[schemas.FaceSelect])
async def get_faces(db: Session = Depends(deps.get_db)) -> Any:
    query = db.query(models.Face).order_by(desc(models.Face.id))
    return paginate(query)


@router.post("/add_face")
async def create_face(file: UploadFile = File(...), name: str = Form(...), phone: str = Form(...),
                      db: Session = Depends(deps.get_db)) -> Any:
    s = time.time()
    you = crud.crud_face.get_by_phone(db, phone=phone)
    image_name = str(uuid.uuid1()) + '.jpg'
    image_b = await file.read()

    if not you:
        file_save_path = os.path.join('./FaceImageData', image_name)
        with open(os.path.join(UPLOAD_DIR, image_name), "wb+") as f:
            f.write(image_b)
        np_arr = np.frombuffer(image_b, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        try:
            bboxes, kpss = Detector.detect(img)
            if bboxes.shape[0] == 0:
                return JSONResponse(status_code=400, content={"error_message": "No Face Detected"})
            else:
                max_row_index = np.argmax(bboxes[:, 4])
                bbox = bboxes[max_row_index]
                box = bbox[:-1].astype(int)
                detect_score = bbox[-1]
                kps = kpss[max_row_index]

                query_feature = Recognizer.feature_extract(img, key_points=kps)
                query_feature_l = query_feature.tolist()[0]
                Logger.info(f"name: {name} | take_times: {round(time.time() - s, 3)}")
                face_in = schemas.FaceCreate(name=name, phone=phone, face_features=pickle.dumps(query_feature_l),
                                             face_image_path=file_save_path)
                crud.crud_face.create_face(db, obj_in=face_in)
                return JSONResponse(status_code=200, content={"message": "Face added"})

        except Exception as e:
            print(e)
            return JSONResponse(status_code=400, content={"error_message": "Invalid Face-Image"})

    else:
        return JSONResponse(status_code=409, content={"message": "Phone already exists"})


@router.delete("/delete_face/{face_id}")
async def delete_face(face_id: int, db: Session = Depends(deps.get_db)) -> Any:
    a = crud.crud_face.delete_face_by_id(db, id=face_id)
    if a:
        return JSONResponse(status_code=200, content={"message": "Face deleted"})


@router.put("/update_face/{face_id}")
async def update_face(face_id: int, name: str = Form(...), phone: str = Form(...),
                      db: Session = Depends(deps.get_db)) -> Any:
    # face = crud.crud_face.get(db, id=face_id)
    # print(face)
    # print(name, phone)
    a = crud.crud_face.update_face_by_id(db, face_id=face_id, name=name, phone=phone)
    if a:
        return JSONResponse(status_code=200, content={"message": "Face updated"})
    else:
        return JSONResponse(status_code=400, content={"message": "Phone already exists"})
