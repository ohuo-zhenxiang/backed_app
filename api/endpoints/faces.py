import base64
import os
import pickle
import time
import uuid
from typing import Any, List

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile, Form, Response, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlalchemy import paginate
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

import crud
import models
import schemas
from api import deps
from api.face_core.ArcFace_extract import ArcFaceOrt
from api.face_core.RetinaFace_detect import RetinaFace
from settings import UPLOAD_DIR

router = APIRouter()
Detector = RetinaFace()
Recognizer = ArcFaceOrt()
faces_logger = logger.bind(name="Faces")


@router.get("/get_faces", response_model=Page[schemas.FaceSelect])
async def get_faces(db: Session = Depends(deps.get_db)) -> Any:
    query = db.query(models.Face).order_by(desc(models.Face.id))
    faces_logger.info("get_faces")
    return paginate(query)


@router.get("/get_face_by_id/{face_id}", response_model=schemas.FaceSelect)
async def get_face_by_id(face_id: int, db: Session = Depends(deps.get_db)) -> Any:
    face = db.query(models.Face).filter(models.Face.id == face_id).first()
    if face:
        return face
    else:
        return Response(status_code=204)


@router.post("/add_face")
async def create_face(file: UploadFile = File(...), name: str = Form(...), phone: str = Form(...),
                      gender: str = Form(...), db: Session = Depends(deps.get_db)) -> Any:
    s = time.time()
    you = crud.crud_face.get_by_phone(db, phone=phone)
    image_name = str(uuid.uuid1()) + '.jpg'
    image_b = await file.read()

    if not you:
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
                file_save_path = os.path.join('./FaceImageData', image_name)
                with open(os.path.join(UPLOAD_DIR, image_name), "wb+") as f:
                    f.write(image_b)
                face_in = schemas.FaceCreate(name=name, phone=phone, face_features=pickle.dumps(query_feature_l),
                                             face_image_path=file_save_path, gender=gender, source='Upload')
                crud.crud_face.create_face(db, obj_in=face_in)
                faces_logger.success(f"AddFace | name: {name} | take_times: {round(time.time() - s, 3)}")
                return JSONResponse(status_code=200, content={"message": "Face added"})

        except Exception as e:
            faces_logger.error(f"AddFace | name: {name} | error: {e}")
            return JSONResponse(status_code=400, content={"error_message": "Invalid Face-Image"})

    else:
        faces_logger.warning(f"AddFace | name: {name} | warning: Phone already exists")
        return JSONResponse(status_code=409, content={"message": "Phone already exists"})


class Base64image(BaseModel):
    base64image: str


@router.post('/detect_face')
async def detect_face(post_image: Base64image):
    faces_list = []
    try:
        image_base64 = post_image.base64image.split(',')[1]
        decoded_image = base64.b64decode(image_base64)
        np_arr = np.fromstring(decoded_image, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        bboxes, kpss = Detector.detect(img)
        if bboxes.shape[0] == 0:
            return Response(status_code=204)
        else:
            for bbox, kps in zip(bboxes, kpss):
                box = bbox[:-1].astype(int)
                detect_score = bbox[-1]
                temp_dict = {
                    "box": box.tolist(),
                    "detect_score": round(float(detect_score)*100, 2),
                    "kps": kps.tolist(),
                }
                faces_list.append(temp_dict)

            faces_logger.success(f"DetectFace | faces_count: {len(faces_list)}")
            return JSONResponse(status_code=200, content={"faces": faces_list, "faces_count": len(faces_list)})
    except Exception as e:
        faces_logger.error(f"DetectFace | error: {e}")
        return JSONResponse(status_code=400, content={"error_message": str(e)})


"""@router.post('/add_snapshot')
async def create_snapshot(post_data: FormSnap, db: Session = Depends(deps.get_db)):
    s = time.time()

    name = post_data.name
    phone = post_data.phone

    image_name = str(uuid.uuid1()) + ".jpg"
    you = crud.crud_face.get_by_phone(db, phone=phone)
    if not you:
        image_base64 = post_data.image.split(',')[1]
        decoded_image = base64.b64decode(image_base64)
        nparr = np.fromstring(decoded_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
                logger.info(f"name: {name} | take_times: {round(time.time() - s, 3)}")
                file_save_path = os.path.join('./FaceImageData', image_name)
                cv2.imwrite(os.path.join(UPLOAD_DIR, image_name), img)

                face_in = schemas.FaceCreate(name=name, phone=phone, face_features=pickle.dumps(query_feature_l),
                                             face_image_path=file_save_path, source='Snapshot')
                crud.crud_face.create_face(db, obj_in=face_in)
                return JSONResponse(status_code=200, content={"message": "Face added"})

        except Exception as e:
            print(e)
            return JSONResponse(status_code=400, content={"error_message": "Invalid Face-Image"})

    else:
        return JSONResponse(status_code=409, content={"message": "Phone already exists"})"""


class FormSnap(BaseModel):
    name: str
    phone: str
    gender: str
    image64: str
    kps: List[list]
    box: List[int]


@router.post('/add_snapshot_face')
async def add_snapshot(post_data: FormSnap, db: Session = Depends(deps.get_db)):
    s = time.time()
    you = crud.crud_face.get_by_phone(db, phone=post_data.phone)
    if not you:
        image_name = str(uuid.uuid1())+'.jpg'
        try:
            image_base64 = post_data.image64.split(',')[1]
            decoded_image = base64.b64decode(image_base64)
            np_arr = np.fromstring(decoded_image, np.uint8)
            raw_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = raw_img[post_data.box[1]:post_data.box[3], post_data.box[0]:post_data.box[2]]

            query_feature = Recognizer.feature_extract(raw_img, key_points=np.array(post_data.kps))
            query_feature_l = query_feature.tolist()[0]
            file_save_path = os.path.join('./FaceImageData', image_name)
            cv2.imwrite(os.path.join(UPLOAD_DIR, image_name), img)

            face_in = schemas.FaceCreate(name=post_data.name, phone=post_data.phone, gender=post_data.gender,
                                         face_features=pickle.dumps(query_feature_l),
                                         face_image_path=file_save_path, source="Snapshot")
            crud.crud_face.create_face(db, obj_in=face_in)
            faces_logger.success(f"AddSnapshotFace | name: {post_data.name} | take_times: {round(time.time()-s, 3)}")
            return JSONResponse(status_code=200, content={"message": "Face added"})
        except Exception as e:
            faces_logger.error(f"AddSnapshotFace | name: {post_data.name} | error: {e}")
            return JSONResponse(status_code=400, content={"error": e})
    else:
        faces_logger.warning(f"AddSnapshotFace | name: {post_data.name} | warning: Phone number already exists")
        return JSONResponse(status_code=409, content={"error": "Phone number already exists"})


@router.delete("/delete_face/{face_id}")
async def delete_face(face_id: int, db: Session = Depends(deps.get_db)) -> Any:
    a = crud.crud_face.delete_face_by_id(db, id=face_id)
    if a:
        faces_logger.success(f"DeleteFace | face_id: {face_id} | deleted")
        return JSONResponse(status_code=200, content={"message": "Face deleted"})
    else:
        faces_logger.warning(f"DeleteFace | face_id: {face_id} | warning: Face not found")
        return JSONResponse(status_code=400, content={"error": "Face not found"})


@router.put("/update_face_without_image/{face_id}")
async def update_face_without_image(face_id: int, name: str = Form(...),
                                    phone: str = Form(...),
                                    gender: str = Form(None),
                                    db: Session = Depends(deps.get_db)) -> Any:
    # face = crud.crud_face.get(db, id=face_id)
    # print(face)
    # print(name, phone)
    a = crud.crud_face.update_face_by_id(db, face_id=face_id, name=name, phone=phone, gender=gender, source='Upload')
    if a:
        faces_logger.success(f"UpdateFaceWithoutImage | face_id: {face_id} | updated")
        return JSONResponse(status_code=200, content={"message": "Face updated"})
    else:
        faces_logger.warning(f"UpdateFaceWithoutImage | face_id: {face_id} | Phone already exists")
        return JSONResponse(status_code=409, content={"message": "Phone already exists"})


@router.put("/update_face_with_image/{face_id}")
async def update_face_with_image(face_id: int, name: str = Form(...), phone: str = Form(...), gender: str = Form(...),
                                 file: UploadFile = File(...),
                                 background_tasks: BackgroundTasks = None,
                                 db: Session = Depends(deps.get_db)) -> Any:
    if db.query(models.Face).filter(models.Face.phone == phone, models.Face.id != face_id).first():
        faces_logger.warning(f"UpdateFaceWithImage | face_id: {face_id} | Phone already exists")
        return JSONResponse(status_code=409, content={"message": "Phone already exists"})

    try:
        image_b = await file.read()
        np_arr = np.frombuffer(image_b, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        bboxes, kpss = Detector.detect(img)
        if bboxes.shape[0] == 0:
            return JSONResponse(status_code=423, content={"error_message": "No Face Detected"})
        else:
            max_row_index = np.argmax(bboxes[:, 4])
            bbox = bboxes[max_row_index]
            box = bbox[:-1].astype(int)
            detect_score = bbox[-1]
            kps = kpss[max_row_index]

            query_feature = Recognizer.feature_extract(img, key_points=kps)
            query_feature_l = query_feature.tolist()[0]

            b = crud.crud_face.get_face_by_id(db, id=face_id)
            old_image_name = b.face_image_path.split('\\')[-1]
            background_tasks.add_task(delete_old_faceImage, old_image_name)
            image_name = str(uuid.uuid1()) + '.jpg'
            file_save_path = os.path.join('./FaceImageData', image_name)
            with open(os.path.join(UPLOAD_DIR, image_name), "wb+") as f:
                f.write(image_b)

            a = crud.crud_face.update_face_by_id2(db, name=name, face_id=face_id, phone=phone, gender=gender,
                                                  source='Upload', face_features=pickle.dumps(query_feature_l),
                                                  face_image_path=file_save_path)
            faces_logger.success(f"UpdateFaceWithImage | face_id: {face_id} | updated")
            return JSONResponse(status_code=200, content={"message": "Face updated"})

    except Exception as e:
        faces_logger.error(f"UpdateFaceWithImage | face_id: {face_id} | error: {e}")
        return JSONResponse(status_code=415, content={"error_message": "Invalid Face-Image"})


def safe_remove_file(file_path):
    max_attempts = 5
    attempts = 0
    while attempts < max_attempts:
        try:
            os.remove(file_path)
            return True
        except OSError:
            attempts += 1
            time.sleep(1)
    return False


async def delete_old_faceImage(image_name):
    a = safe_remove_file(os.path.join(UPLOAD_DIR, image_name))
    if a:
        faces_logger.success(f"DeleteOldFaceImage | image_name: {image_name} | deleted")
    else:
        faces_logger.error(f"DeleteOldFaceImage failed")

