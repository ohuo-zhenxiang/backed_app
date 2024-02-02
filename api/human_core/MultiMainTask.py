import json
import os
import time
from datetime import datetime
from typing import List

import cv2
from func_timeout import func_set_timeout
from func_timeout.exceptions import FunctionTimedOut

from api.human_core.Human_detect import HumanDetect_V2
from api.human_core.Pose_detect import PoseDetect_V2
from api.human_core.Smoking_detect import SmokingCallingDetect_V2
from api.human_core.utils import Profile
from db.session import SessionLocal
from models import HumanRecord, HumanTask
from redis_module import RedisModule
from settings import LOGGING_DIR

task2model = {
    'smoke': SmokingCallingDetect_V2,
    'phone': SmokingCallingDetect_V2,
    'pose': PoseDetect_V2,
}


def multi_step_2(ex_tasks: List[str]):
    return list({task2model[task] for task in ex_tasks})


def snap_multi_analysis_core(task_token: str, task_ex: List[str], capture_path: str | int, save_fold: str, _logger):
    @func_set_timeout(3)
    def capture_init(path):
        capture = cv2.VideoCapture(path)
        return capture

    start_time = datetime.now().replace(microsecond=0)

    # 增record_status和error_info记录任务状态和错误信息
    record_status, task_result, human_count, record_image_path = '', {}, 0, ''

    sss = time.time()
    # 先实例化人体检测，加载模型，热干面的面
    human_detect = HumanDetect_V2()
    persons_res = []

    try:
        s = time.time()
        cap = capture_init(capture_path)
        _logger.info(f"capture init success, take times: {(time.time() - s):.3f}")
    except FunctionTimedOut as e:
        _logger.error("can't init capture")
        record_status, error_info = "Record Failed", "can't init capture"
        return record_status, error_info, task_result, start_time, human_count, record_image_path
    else:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            _logger.error("can't read frame")
            record_status, error_info = "Record Failed", "can't read frame"
            return record_status, error_info, task_result, start_time, human_count, record_image_path
        else:
            img0 = frame.copy()
            # 相对地址
            record_image_path = os.path.join('./TaskRecord', f"{task_token}",
                                             f"{start_time.strftime('%Y-%m-%d %H-%M-%S')}.jpg")

            try:
                dt = [Profile()] * 3

                """person detect"""
                with dt[0]:
                    person_res = human_detect.forward(img0)
                _logger.success(
                    f"person detect success, take times: {dt[0].dt:.3f}, person-count: {person_res.shape[0]}")

                if person_res.shape[0] != 0:
                    persons_res = human_detect.record_result(person_res)

                    # init ex- model()、model().forward、model().record_res
                    for model in multi_step_2(task_ex):
                        with dt[1]:
                            model = model()
                            res = model.forward(person_res, img0)
                            persons_res = model.record_result(res, persons_res)
                        _logger.success(f"ex-model: {model.__class__.__name__} success, take times: {dt[1].dt:.3f}")

                    # pprint([x.model_dump() for x in persons_res])
                cv2.imwrite(os.path.join(save_fold, f"{start_time.strftime('%Y-%m-%d %H-%M-%S')}.jpg"), frame)
                record_status, error_info = 'Record Completed', ''
                human_count = len(persons_res)
                task_result = {
                    "humans": [x.model_dump() for x in persons_res],
                    "humans_count": human_count,
                    "record_status": record_status
                }
                _logger.success(
                    f"Record Completed; {len(persons_res)} humans detected | take times: {(time.time() - sss):.3f}")
                return record_status, error_info, task_result, start_time, human_count, record_image_path

            except Exception as e:
                print(e)
                record_status, error_info = "Task Failed", ''
                task_result = {"humans": [], "humans_count": 0}
                _logger.error(f"Task Failed | error: {e}")
                return record_status, error_info, task_result, start_time, human_count, record_image_path
            finally:
                cap.release()
                _logger.complete()


def SnapMultiAnalysis(task_token: str, task_ex: List[str], capture_path: str | int, save_fold: str):
    """
    SnapMultiAnalysis
    :param task_token: 任务token
    :param task_ex: 扩展任务列表，e.i. ["smoke", "phone", "pose"]
    :param capture_path: 流地址
    :param save_fold: 保存地址
    :return:
    """
    from loguru import logger
    logger.remove()  # 清除控制台的log，不然没法存到log文件，会报错
    task_logger = logger.bind(task_name=f"task{task_token}")
    task_logger.add(f"{LOGGING_DIR}/HUMAN_TASK_{task_token}.log", rotation="200 MB", retention="30 days",
                    encoding="utf-8", enqueue=True)

    record_status, error_info, task_result, start_time, human_count, record_image_path = snap_multi_analysis_core(
        task_token,
        task_ex,
        capture_path,
        save_fold,
        task_logger)
    completed_time = datetime.now().replace(microsecond=0)
    db = SessionLocal()
    try:
        db_obj = HumanRecord(start_time=start_time,
                             human_count=human_count,
                             record_info=json.dumps(task_result),
                             task_token=task_token,
                             completed_time=completed_time,
                             record_image_path=record_image_path,
                             record_status=record_status,
                             error_info=error_info)
        db.add(db_obj)
        db.commit()

        # redis publish
        with RedisModule() as R:
            R.publish(f"{task_token}",
                      json.dumps({
                          "status": record_status, "record_info": task_result,
                          "start_time": start_time.strftime('%Y-%m-%d %H-%M-%S'),
                          "completed_time": completed_time.strftime('%Y-%m-%d %H-%M-%S'),
                          "human_count": human_count,
                          "record_image_path": record_image_path,
                          "record_status": record_status,
                          "error_info": error_info,
                      }))
    finally:
        db.close()


def MultiUpdateStatus(task_token: str, status: str):
    """
    更新任务状态
    :param task_token:
    :param status:
    :return:
    """
    db = SessionLocal()
    try:
        db.query(HumanTask).filter(HumanTask.task_token == task_token).update({"status": status})
        db.commit()
    finally:
        db.close()


if __name__ == '__main__':
    task_extends = ['smoke', 'phone', 'pose']
    model_list = multi_step_2(task_extends)

    from loguru import logger

    cap_path = 'rtsp://192.168.130.182:554'
    snap_multi_analysis_core(task_token='test', task_ex=task_extends, capture_path=cap_path, save_fold='',
                             _logger=logger)
