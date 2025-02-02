import json
import os.path
import time
from datetime import datetime
from typing import List

import cv2
from func_timeout import func_set_timeout
from func_timeout.exceptions import FunctionTimedOut

from api.face_core.Feature_retrieval import Retrieval
from api.face_core.RetinaFace_detect import RetinaFace
from api.face_core.SilentFaceAntiSpoofing import SilentFaceAntiSpoofing
from db.session import SessionLocal
from models import Record, Task
from redis_module import RedisModule
from settings import LOGGING_DIR, REC_THRESHOLD


# 数据库和log文件双存储error情况
def snap_analysis(task_token: str, ex_detect: List[str], capture_path: str, save_fold: str, _logger):
    @func_set_timeout(3)
    def capture_init(path):
        capture = cv2.VideoCapture(path)
        return capture

    start_time = datetime.now().replace(microsecond=0)
    sss = time.time()

    # 头部的两个模型初始化
    R = Retrieval(task_id=task_token)
    detector = RetinaFace()

    # 用record_status和error_info记录任务状态和错误信息
    record_status, error_info, task_result, face_count, record_image_path, record_names = '', '', {}, 0, '', []
    try:
        s = time.time()
        cap = capture_init(capture_path)
        _logger.info(f"capture init success, take times: {round(time.time() - s, 3)}s")
    except FunctionTimedOut as e:
        # 摄像头连接超时
        _logger.error("can't init capture")
        record_status, error_info = "Record Failed", "can't init capture"
        return record_status, error_info, task_result, start_time, face_count, record_image_path, record_names
    else:
        ret, frame = cap.read()
        if not ret:
            # 摄像头读取失败
            cap.release()
            _logger.error("can't read frame")
            record_status, error_info = "Record Failed", "can't read frame"
            return record_status, error_info, task_result, start_time, face_count, record_image_path, record_names
        else:
            img = frame.copy()

            bboxes_5, kpss = detector.detect(img)
            # 存相对路径，前端再根据后端服务地址去拼完整路径
            record_image_path = os.path.join('./TaskRecord', f"{task_token}",
                                             f"{start_time.strftime('%Y-%m-%d %H-%M-%S')}.jpg")

            try:
                faces_list = []
                if bboxes_5.shape[0] > 0:
                    for i, bbox in enumerate(bboxes_5):
                        temp_dict = {}
                        box = bbox[:-1].astype(int)
                        detect_score = bbox[-1]
                        kps = kpss[i]

                        face_reg_name_index, face_reg_similarity = R.run_retrieval(img, kps)
                        if face_reg_similarity > REC_THRESHOLD:
                            label = str(R.names[face_reg_name_index])
                            label_id = str(R.fids[face_reg_name_index])
                            record_names.append(label)
                            # print(face, label)
                        else:
                            label = "UNK"
                            label_id = "UNK"

                        if 'SFAS' in ex_detect:
                            sfas = SilentFaceAntiSpoofing()
                            sfas_label, sfas_trueness = sfas.forward(box=box, img=img)
                        else:
                            sfas_label = 1
                            sfas_trueness = 1.0

                        temp_dict.update(
                            {
                                "box": box.tolist(),
                                "detect_score": round(float(detect_score), 3),
                                "kps": kps.tolist(),
                                "label": label,
                                "label_id": label_id,
                                "is_fake": False if sfas_label == 1 else True,
                                "sfas_trueness": round(float(sfas_trueness), 3)
                            }
                        )
                        faces_list.append(temp_dict)
                        '''230907 不再后端画框，返原图和坐标前端绘制'''
                        # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=2)
                        # cv2.putText(img, f"{label}", (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        #             0.8, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(save_fold, f"{start_time.strftime('%Y-%m-%d %H-%M-%S')}.jpg"), img)
                record_status, error_info = "Record Completed", ""
                task_result = {"faces": faces_list, "faces_count": len(faces_list)}
                face_count = len(faces_list)
                _logger.success(
                    f"Record Completed; {face_count} faces detected | take times: {(time.time() - sss):}.2fs")
                return record_status, error_info, task_result, start_time, face_count, record_image_path, record_names
            except Exception as e:
                print(e)
                record_status, error_info = "Record Failed", e
                task_result = {"faces": [], "faces_count": 0}
                _logger.error(f"Record Failed | error: {e}")
                return record_status, error_info, task_result, start_time, face_count, record_image_path, record_names
            finally:
                cap.release()
                _logger.complete()
                # print('-------', time.time() - sss)


def SnapAnalysis(task_token: str, ex_detect: List[str], capture_path: str, save_fold: str):
    """
    SnapAnalysis
    :param task_token: 任务token
    :param ex_detect: 扩展的检测，e.i. ['SFAS']
    :param capture_path: 流地址
    :param save_fold: 保存路径
    :return:
    """
    from loguru import logger
    logger.remove()
    task_logger = logger.bind(task_name=f"task{task_token}")
    task_logger.add(f"{LOGGING_DIR}/FACE_TASK_{task_token}.log",
                    rotation="200 MB",
                    retention="30 days",
                    encoding="utf-8",
                    enqueue=True)

    record_status, error_info, task_result, start_time, face_count, record_image_path, record_names = snap_analysis(
        task_token, ex_detect, capture_path, save_fold, task_logger)
    completed_time = datetime.now().replace(microsecond=0)
    db = SessionLocal()
    try:
        db_obj = Record(start_time=start_time, face_count=face_count, record_info=json.dumps(task_result),
                        task_token=task_token, completed_time=completed_time, record_image_path=record_image_path,
                        record_names=record_names, record_status=record_status, error_info=error_info)
        db.add(db_obj)
        db.commit()
        # db.refresh(db_obj)

        # redis publish
        with RedisModule() as R:
            R.publish(f"{task_token}",
                      json.dumps({
                          "status": record_status, "record_info": task_result,
                          "start_time": start_time.strftime('%Y-%m-%d %H-%M-%S'),
                          "completed_time": completed_time.strftime('%Y-%m-%d %H-%M-%S'),
                          "face_count": face_count,
                          "record_image_path": record_image_path, "record_names": record_names,
                          "record_status": record_status, "error_info": error_info,
                      }))
    finally:
        db.close()


def UpdateStatus(task_token: str, status: str):
    """
    更新任务状态
    :param task_token: 任务token
    :param status: 任务状态
    :return:
    """
    # print(f"task_token: {task_token}, status: {status}")
    db = SessionLocal()
    try:
        db.query(Task).filter(Task.task_token == task_token).update({"status": status})
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    pass
