import os.path

from settings import TASK_RECORD_DIR
from .RetinaFace_detect import RetinaFace
from .ArcFace_extract import ArcFaceOrt

from func_timeout import func_set_timeout
from pprint import pprint
from datetime import datetime
import cv2
import time
import base64
from logger_module import Logger



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

    @func_set_timeout(3)
    def capture_init(path):
        capture = cv2.VideoCapture(path)
        return capture

    detector_path = 'api/face_core/model/det_10g.onnx'
    sss = time.time()
    detector = RetinaFace(detector_path)
    print("load---", time.time() - sss)
    cap = None
    task_status, task_result = '', {}
    try:
        s = time.time()
        cap = capture_init(capture_path)
        Logger.info(f"capture init success, take times: {round(time.time() - s, 3)}s")
    except Exception as e:
        Logger.error("can't init capture")
        task_status = "Capture Error"
        print(e)
        # return task_status, task_result
    if cap is not None:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            Logger.error("can't read frame")
            task_status = "Capture Error"
            # return task_status, task_result
        else:
            execution_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            img = frame.copy()

            # cv2.imwrite(f"{raw_save_path}/{execution_time}.jpg", img)

            bboxes_5, kpss = detector.detect(img)

            try:
                faces_list = []
                if bboxes_5.shape[0] > 0:
                    for i, bbox in enumerate(bboxes_5):
                        box = bbox[:-1].astype(int)
                        detect_score = bbox[-1]
                        kps = kpss[i]
                        temp_dict = {
                            "box": box,
                            "detect_score": detect_score,
                            "kps": kps}
                        faces_list.append(temp_dict)
                        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=2)

                cv2.imwrite(f"{save_fold}/{execution_time}.jpg", img)
                task_status = "Task Completed"
                task_result = {"faces": faces_list, "task_status": task_status,
                               "image_path": f"{save_fold}/{execution_time}.jpg",
                               "faces_count": len(faces_list)}
                # return task_status, task_result
            except Exception as e:
                task_status = "Task Failed"
                task_result = {"faces": [], "task_status": task_status,
                               "image_path": f"{save_fold}/{execution_time}.jpg",
                               "faces_count": 0}
                print(e)
                # return task_status, task_result
            finally:
                print(task_status)
                print(task_result.get("faces_count", 'error'))
                cap.release()
                print('-------', time.time() - sss)


if __name__ == "__main__":
    s = time.time()
    Detector = RetinaFace()
    Recognizer = ArcFaceOrt()
    print(f"load detect model time: {time.time() - s}")

    s = time.time()
    task_status, task_result = SnapAnalysis(task_token="test",
                                            capture_path='rtsp://192.168.130.182:554',
                                            detector=Detector,
                                            recognizer=Recognizer)
    print(task_status)
    pprint(task_result)

    print(time.time() - s)
