import json
import os
import time
from datetime import datetime

import cv2
from func_timeout import func_set_timeout

from db.session import SessionLocal, async_session
from models import HumanTask, HumanRecord
from redis_module import RedisModule
from settings import LOGGING_DIR
from .Smoking_detect import SmokingCallingDetect
from .Human_detect import Profile


def snap_sandc_analysis_core(task_token: str, capture_path: str | int, save_fold: str, _logger):
    @func_set_timeout(3)
    def capture_init(path):
        capture = cv2.VideoCapture(path)
        return capture

    start_time = datetime.now().replace(microsecond=0)
    cap = None
    task_status, task_result, human_count, record_image_path = '', {}, 0, ''

    sss = time.time()
    dzDetector = SmokingCallingDetect()
    _logger.info(f"DZ detector init success, take times: {(time.time() - sss):.3f}")


    try:
        s = time.time()
        cap = capture_init(capture_path)
        _logger.info(f"capture init success, take times: {(time.time() -s):.3f}")
    except Exception as e:
        _logger.error("can't init capture")
        task_status = "Capture Error"
        return task_status, task_result, start_time, human_count, record_image_path

    if cap is not None:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            _logger.error("can't read frame")
            task_status = "Capture Error"
            return task_status, task_result, start_time, human_count, record_image_path
        else:
            img0 = frame.copy()
            record_image_path = os.path.join('./TaskRecord', f"{task_token}",
                                             f"{start_time.strftime('%Y-%m-%d %H-%M-%S')}.jpg")
            try:
                dt = (Profile(), Profile())

                """person detect"""
                with dt[0]:
                    person_res = dzDetector.human_detect(img0)
                _logger.success(f"person detect success, take times: {dt[0].dt:.3f}, person-count: {person_res.shape[0]}")

                persons_res = []
                if person_res.shape[0] != 0:
                    """sandc detect"""
                    with dt[1]:
                        FRW_res = dzDetector.smoking_calling_detect(person_res[:, :5], img0)
                        for i, person in enumerate(person_res):
                            person_box = [int(x) for x in person[:4]]
                            person_res = {
                                'person_id': i,
                                'person_box': person_box,
                                'person_score': float(person[4]),
                                'person_behaviors': {"smoking": [], "calling": []}
                            }
                            behavior_names = dzDetector.names_2
                            if FRW_res[i].shape[0] != 0:
                                for j, frw in enumerate(FRW_res[i]):
                                    bbox = [int(bb) for bb in frw[:4]]
                                    class_id = int(frw[-1])
                                    behavior_res = {
                                        'behavior_box': [eix + wai for eix, wai in zip(person_box[:2] * 2, bbox)],
                                        'behavior_type': behavior_names[class_id],
                                        'behavior_score': float(frw[4]),
                                    }
                                    if class_id == 0:
                                        person_res['person_behaviors']['smoking'].append(behavior_res)
                                    elif class_id == 1:
                                        person_res['person_behaviors']['calling'].append(behavior_res)
                            persons_res.append(person_res)

                cv2.imwrite(os.path.join(save_fold, f"{start_time.strftime('%Y-%m-%d %H-%M-%S')}.jpg"), frame)
                task_status, human_count = "Task Completed", len(persons_res)
                task_result = {"humans": persons_res, "humans_count": human_count, "task_status": task_status}
                _logger.success(f"Task Completed; {len(persons_res)} humans detected | take times: {(time.time()-sss):.3f}")
            except Exception as e:
                print(e)
                task_status = "Task Failed"
                task_result = {"humans": [], "humans_count": 0, "task_status": task_status}
                _logger.error(f"Task Failed | error: {e}")

            finally:
                cap.release()
                _logger.complete()
                return task_status, task_result, start_time, human_count, record_image_path


def SnapSandCAnalysis(task_token: str, capture_path: str, save_fold: str):
    """
    SnapSandCAnalysis
    :param task_token: 任务token
    :param capture_path: 流地址
    :param save_fold: 保存路径
    :return:
    """
    from loguru import logger
    logger.remove()
    task_logger = logger.bind(task_name=f"task{task_token}")
    task_logger.add(f"{LOGGING_DIR}/HUMAN_TASK_{task_token}.log", rotation="200 MB", retention="30 days",
                    encoding="utf-8", enqueue=True)

    task_status, task_result, start_time, human_count, record_image_path = snap_sandc_analysis_core(task_token, capture_path, save_fold, task_logger)
    completed_time = datetime.now().replace(microsecond=0)
    db = SessionLocal()
    try:
        db_obj = HumanRecord(start_time=start_time, human_count=human_count, record_info=json.dumps(task_result),
                             task_token=task_token, completed_time=completed_time, record_image_path=record_image_path)
        db.add(db_obj)
        db.commit()
    finally:
        db.close()


def SCUpdateStatus(task_token: str, status: str):
    """
    更新任务状态
    :param task_token: 任务token
    :param status: 任务状态
    :return:
    """
    db = SessionLocal()
    try:
        db.query(HumanTask).filter(HumanTask.task_token == task_token).update({"status": status})
        db.commit()
    finally:
        db.close()

