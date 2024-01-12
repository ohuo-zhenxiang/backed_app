import json
import os.path
import time
from datetime import datetime

import cv2
from func_timeout import func_set_timeout

from db.session import SessionLocal
from models import HumanTask, HumanRecord
from redis_module import RedisModule
from settings import LOGGING_DIR
from .Human_detect import HumanDetect


def snap_human_analysis_core(task_token: str, capture_path, save_fold: str, _logger):
    @func_set_timeout(3)
    def capture_init(path):
        capture = cv2.VideoCapture(path)
        return capture

    start_time = datetime.now().replace(microsecond=0)
    sss = time.time()
    humanDetector = HumanDetect()
    cap = None
    task_status, task_result, human_count, record_image_path = '', {}, 0, ''

    try:
        s = time.time()
        cap = capture_init(capture_path)
        _logger.info(f"capture init success, take times: {(time.time() - s):.3f}")
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
            img = frame.copy()
            record_image_path = os.path.join('./TaskRecord', f"{task_token}",
                                             f"{start_time.strftime('%Y-%m-%d %H-%M-%S')}.jpg")
            try:
                det = humanDetector.human_detect(frame)
                humans_list = []
                if det is not None and len(det) > 0:
                    for i, (*xyxy, conf, cls) in enumerate(reversed(det)):
                        confidence = float(conf)
                        confidence_str = f'{confidence:.2f}'
                        temp_dict = {
                            "person_box": [int(x) for x in xyxy],
                            "person_score": confidence_str,
                            "person_id": i,
                        }
                        humans_list.append(temp_dict)

                cv2.imwrite(os.path.join(save_fold, f"{start_time.strftime('%Y-%m-%d %H-%M-%S')}.jpg"), img)
                task_status = "Task Completed"
                task_result = {"humans": humans_list, "humans_count": len(humans_list), "task_status": task_status}
                human_count = len(humans_list)
                _logger.success(
                    f"Task Completed; {len(humans_list)} humans detected | take times: {(time.time() - sss):.2f}")
            except Exception as e:
                print(e)
                task_status = "Task Failed"
                task_result = {"humans": [], "humans_count": 0, "task_status": task_status}
                _logger.error(f"Task Failed | error: {e}")
            finally:
                cap.release()
                _logger.complete()
                return task_status, task_result, start_time, human_count, record_image_path


def SnapHumanAnalysis(task_token: str, capture_path: str, save_fold: str):
    """
    SnapHumanAnalysis
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

    task_status, task_result, start_time, human_count, record_image_path = snap_human_analysis_core(task_token,
                                                                                                    capture_path,
                                                                                                    save_fold,
                                                                                                    task_logger)
    completed_time = datetime.now().replace(microsecond=0)
    db = SessionLocal()
    try:
        db_obj = HumanRecord(start_time=start_time, human_count=human_count, record_info=json.dumps(task_result),
                             task_token=task_token, completed_time=completed_time, record_image_path=record_image_path)
        db.add(db_obj)
        db.commit()
        # db.refresh(db_obj)

        # redis publish
        with RedisModule() as R:
            R.publish(f"{task_token}",
                      json.dumps({"status": task_status, "record_info": task_result,
                                  "start_time": start_time.strftime('%Y-%m-%d %H-%M-%S'),
                                  "completed_time": completed_time.strftime('%Y-%m-%d %H-%M-%S'),
                                  "human_count": human_count,
                                  "record_image_path": record_image_path}))

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
        db.query(HumanTask).filter(HumanTask.task_token == task_token).update({"status": status})
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    pass
