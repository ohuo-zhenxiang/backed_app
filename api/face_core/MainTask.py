import os.path
import json
from settings import TASK_RECORD_DIR

from .Feature_retrieval import Retrieval
from .RetinaFace_detect import RetinaFace
from .ArcFace_extract import ArcFaceOrt
from db.session import SessionLocal
from models import Record, Task

from func_timeout import func_set_timeout
from pprint import pprint
from datetime import datetime
import cv2
import time
import base64
from logger_module import Logger


@func_set_timeout(3)
def capture_init(path):
    capture = cv2.VideoCapture(path)
    return capture


def snap_analysis(task_token: str, capture_path: str, save_fold: str):
    start_time = datetime.now().replace(microsecond=0)
    sss = time.time()
    R = Retrieval(task_id=task_token)
    detector = RetinaFace()
    cap = None
    task_status, task_result, face_count, record_image_path = '', {}, 0, ''
    try:
        s = time.time()
        cap = capture_init(capture_path)
        Logger.info(f"capture init success, take times: {round(time.time() - s, 3)}s")
    except Exception as e:
        Logger.error("can't init capture")
        task_status = "Capture Error"
        return task_status, task_result, start_time, face_count, record_image_path
    if cap is not None:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            Logger.error("can't read frame")
            task_status = "Capture Error"
            return task_status, task_result, start_time, face_count
        else:
            img = frame.copy()

            bboxes_5, kpss = detector.detect(img)
            record_image_path = os.path.join('./TaskRecord', f"{task_token}",
                                             f"{start_time.strftime('%Y-%m-%d %H-%M-%S')}.jpg")

            try:
                faces_list = []
                if bboxes_5.shape[0] > 0:
                    for i, bbox in enumerate(bboxes_5):
                        box = bbox[:-1].astype(int)
                        detect_score = bbox[-1]
                        kps = kpss[i]

                        face_reg_name_index, face_reg_similarity = R.run_retrieval(img, kps)
                        if face_reg_similarity > 98:
                            label = str(R.names[face_reg_name_index])
                            label_id = str(R.fids[face_reg_name_index])
                            # print(face, label)
                        else:
                            label = "Unknown"
                            label_id = "Unknown"
                        temp_dict = {
                            "box": box.tolist(),
                            "detect_score": round(float(detect_score), 3),
                            "kps": kps.tolist(),
                            "label": label,
                            "label_id": label_id}
                        faces_list.append(temp_dict)
                        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=2)
                        cv2.putText(img, f"{label}", (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 255, 0), 2)

                cv2.imwrite(os.path.join(save_fold, f"{start_time.strftime('%Y-%m-%d %H-%M-%S')}.jpg"), img)
                print('????', time.time() - sss)
                task_status = "Task Completed"
                task_result = {"faces": faces_list, "task_status": task_status,
                               "faces_count": len(faces_list)}
                face_count = len(faces_list)
                return task_status, task_result, start_time, face_count, record_image_path
            except Exception as e:
                Logger.error(e)
                task_status = "Task Failed"
                task_result = {"faces": [], "task_status": task_status,
                               "faces_count": 0}
                return task_status, task_result, start_time, face_count, record_image_path
            finally:
                cap.release()
                print('-------', time.time() - sss)


def SnapAnalysis(task_token: str, capture_path: str, save_fold: str):
    """
    SnapAnalysis
    :param task_token: 任务token
    :param capture_path: 流地址
    :param detector: RetinaFace的实例对象
    :param recognizer: ArcFaceOrt的实例对象
    :param feature_path: 特征表
    :return:
    """
    task_status, task_result, start_time, face_count, record_image_path = snap_analysis(task_token, capture_path,
                                                                                        save_fold)
    completed_time = datetime.now().replace(microsecond=0)
    db = SessionLocal()
    try:
        db_obj = Record(start_time=start_time, face_count=face_count, record_info=json.dumps(task_result),
                        task_token=task_token, completed_time=completed_time, record_image_path=record_image_path)
        db.add(db_obj)
        db.commit()
    finally:
        db.close()


def UpdateStatus(task_token: str, status: str):
    """
    更新任务状态
    :param task_token: 任务token
    :param status: 任务状态
    :return:
    """
    db = SessionLocal()
    try:
        db.query(Task).filter(Task.task_token == task_token).update({"status": status})
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    pass
