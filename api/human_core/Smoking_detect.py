import contextlib
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
from numba import njit
from Human_detect import Profile

from settings import MODEL_DIR


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current_shape[height, width], new_shape(h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # (width-ratios, height-ratios)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # (h, w)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # (w-padding, h-padding)
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # 原图居中，填充四周
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 用（114,114,114）灰色填充
    return im, ratio, (dw, dh)


def preprocess(im0: np.ndarray):
    im = letterbox(im0)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = im.astype(np.float32)
    im /= 255.
    if len(im.shape) == 3:
        im = im[None]
    im = np.ascontiguousarray(im)
    return im


@njit()
def xyxy2xywh(x):
    # Convert n x 4 boxes from [x ,y, x, y] to [x, y, w, h] where xy=top-left, w=width, h=height
    y = np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # w
    y[..., 3] = x[..., 3] - x[..., 1]  # h
    return y


@njit()
def xywh2xyxy(x:np.ndarray):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def np_non_max_suppression(predictions,
                           img1_shape,
                           img0_shape,
                           conf_thres=0.25,
                           iou_thres=0.45,
                           classes=None,
                           agnostic=False,
                           multi_label=False,
                           labels=(),
                           max_det=300,
                           ):
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]
    bs = predictions.shape[0]
    nc = predictions.shape[2] - 5
    xc = predictions[..., 4] > conf_thres
    max_wh = 7680
    time_limit = 0.5 + 0.5 * bs
    multi_label &= nc > 1

    mi = 5 + nc  # mack start index
    output = [np.zeros((0, 6), dtype=np.float32)] * bs
    for xi, x in enumerate(predictions):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        conf = np.max(x[:, 5:mi], axis=1, keepdims=True)
        j = np.argmax(x[:, 5:mi], axis=1, keepdims=True)
        x = np.concatenate((box, conf, j.astype(np.float32), mask), axis=1)[conf.flatten() > conf_thres]

        if classes is not None:
            classes = np.array(classes)
            a = np.any((x[:, 5:6] == classes[None, :]), axis=1)
            x = x[a]

        n = x.shape[0]
        if not n:
            continue
        # x = x[np.argsort(x[:, 4])[::-1]][:max_nms]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        x[:, :4] = scale_boxes(img1_shape[xi], x[:, :4], img0_shape[xi]).round()
        boxes, scores = x[:, :4] + c, x[:, 4]
        indices = cv2.dnn.NMSBoxes(xyxy2xywh(boxes), scores, conf_thres, iou_thres)
        output[xi] = x[indices]

    return output


def get_multi_inputs(bboxes: np.ndarray, ori_img: np.ndarray, output_size=320):
    """
    @parma:
        bboxes: np.ndarray, shape=(N, 5), [x1, y1, x2, y2, score]
        ori_img: np.ndarray, shape=(H, W, C)
        output_size: int, 下个模型的输入尺寸、
    @return:
        resized_rois: np.ndarray, shape=(N, 3, output_size, output_size)
        boxes
    """
    resized_n_rois = np.zeros((bboxes.shape[0], 3, output_size, output_size), dtype=np.uint8)  # N = bboxes.shape[0]
    for i, b in enumerate(bboxes):
        x1, y1, x2, y2 = b[:4].astype(np.int32)
        roi = ori_img[y1:y2, x1:x2, :]
        resized_roi, _, _ = letterbox(roi, output_size)
        resized_n_rois[i] = resized_roi.transpose((2, 0, 1))[::-1]  # img. HWC => CHW, BGR => RGB

    resized_n_rois = np.ascontiguousarray(resized_n_rois)
    # print('----------------', resized_n_rois.shape)
    return resized_n_rois


class SmokingCallingDetect:
    def __init__(self,
                 weight1=os.path.join(MODEL_DIR, "yolov5x.onnx"),
                 weight2=os.path.join(MODEL_DIR, "best_dynamic_simplified.onnx"),
                 is_gpu=False):
        self.weight_1 = weight1
        self.weight_2 = weight2
        self.conf_thres = 0.45
        self.iou_thres = 0.45
        self.device = torch.device('cuda' if torch.cuda.is_available() and is_gpu else 'cpu')
        self.providers = ['CPUExecutionProvider', 'CUDAExecutionProvider'] if is_gpu else ['CPUExecutionProvider']

        # person onnx
        self.session_1 = ort.InferenceSession(self.weight_1, providers=self.providers)
        self.output_names_1 = [x.name for x in self.session_1.get_outputs()]
        self.meta_1 = self.session_1.get_modelmeta().custom_metadata_map
        if 'stride' in self.meta_1:
            self.stride_1, self.names_1 = int(self.meta_1['stride']), eval(self.meta_1["names"])

        # smoking and calling onnx
        self.session_2 = ort.InferenceSession(self.weight_2, providers=self.providers)
        self.output_names_2 = None
        self.ip_h_2, self.ip_w_2 = self.session_2.get_inputs()[0].shape[2:]
        self.meta_2 = self.session_2.get_modelmeta().custom_metadata_map
        self.stride_2, self.names_2 = self.meta_2['stride'], eval(self.meta_2['names'])
        if isinstance(self.names_2, dict):
            self.classes_2, self.names_2 = list(self.names_2.keys()), list(self.names_2.values())
        else:
            self.classes_2, self.names_2 = [i for i in range(len(self.names_2))], self.names_2

    def warmup(self):
        im1 = np.ones((1, 3, 640, 640), dtype=np.float32)
        im2 = np.random.randn(4, 3, self.ip_h_2, self.ip_w_2).astype(np.float32)
        for _ in range(2):
            dz = self.session_1.run(self.output_names_1, {self.session_1.get_inputs()[0].name: im1})
            wy = self.session_2.run(self.output_names_2, {self.session_2.get_inputs()[0].name: im2})
        return dz[0].shape, wy[0].shape

    def human_detect(self, im0: np.ndarray):
        """preprocess"""
        im = preprocess(im0)

        """Inference"""
        pred = self.session_1.run(self.output_names_1, {self.session_1.get_inputs()[0].name: im})

        """postprocess"""
        res = np_non_max_suppression(pred,
                                     img1_shape=[im.shape[2:]],
                                     img0_shape=[im0.shape],
                                     conf_thres=self.conf_thres,
                                     iou_thres=self.iou_thres,
                                     classes=[0, ], )

        return res[0]

    def smoking_calling_detect(self, person_res: np.ndarray, im0: np.ndarray):
        """preprocess"""
        n_inputs = get_multi_inputs(person_res[:, :5], im0, self.ip_h_2)
        n_inputs = n_inputs.astype(np.float32)
        n_inputs /= 255.

        """Inference"""
        sandc_pred = self.session_2.run(None, {self.session_2.get_inputs()[0].name: n_inputs})[0]

        """postprocess"""
        person_boxes_hw = np.column_stack((person_res[:, 3]-person_res[:, 1], person_res[:, 2]-person_res[:, 0]))
        sandc_res = np_non_max_suppression(sandc_pred,
                                           img1_shape=[(self.ip_h_2, self.ip_w_2)] * n_inputs.shape[0],
                                           img0_shape=person_boxes_hw,
                                           conf_thres=0.5,
                                           iou_thres=0.45,
                                           classes=[0, 1],)

        return sandc_res

    def FRW_detect(self, im0):
        dt = [Profile()] * 3

        """person detect"""
        with dt[0]:
            person_res = self.human_detect(im0)
            print(person_res.shape)

        persons_res = []
        if person_res.shape[0] == 0:
            return None
        else:
            """sandc detect"""
            with dt[1]:
                FRW_res = self.smoking_calling_detect(person_res[:, :5], im0)
                for i, person in enumerate(person_res):
                    person_box = [int(x) for x in person[:4]]
                    person_res = {
                        'person_id': i,
                        'person_box': person_box,
                        'person_score': float(person[4]),
                        'person_behaviors': {"smoking": [], "calling": []}
                    }
                    behavior_names = self.names_2
                    if FRW_res[i].shape[0] != 0:
                        for j, frw in enumerate(FRW_res[i]):
                            bbox = [int(bb) for bb in frw[:4]]
                            class_id = int(frw[-1])
                            behavior_res = {
                                'behavior_box': [eix+wai for eix, wai in zip(person_box[:2]*2, bbox)],
                                'behavior_type': behavior_names[class_id],
                                'behavior_score': float(frw[4]),
                            }
                            if class_id == 0:
                                person_res['person_behaviors']['smoking'].append(behavior_res)
                            elif class_id == 1:
                                person_res['person_behaviors']['calling'].append(behavior_res)
                    persons_res.append(person_res)

            return persons_res


if __name__ == '__main__':
    im = cv2.imread(r'D:\ZHIHUI\Full_stack\Fastapi_Vue\main'
                    r'\backend\app\TaskRecord\8fe60e15-ebc3-456d-9a98-d305a2b839da'
                    r'\2023-10-27 13-37-35.jpg')
    DZ = SmokingCallingDetect()
    a, b = DZ.warmup()
    print(a, b)

    print("------------------------")
    c = DZ.FRW_detect(im.copy())
    from pprint import pprint
    pprint(c)
    from ultralytics.utils.plotting import Annotator

    annotator = Annotator(im, line_width=2)
    for person in c:
        annotator.box_label(person['person_box'], label='person', color=[0, 255, 0])
        if len(person['person_behaviors']['smoking']) > 0:
            for dz in person['person_behaviors']['smoking']:
                annotator.box_label(dz['behavior_box'], label='smoking', color=[0, 0, 255])
        if len(person['person_behaviors']['calling'])> 0:
            for dz in person['person_behaviors']['calling']:
                annotator.box_label(dz['behavior_box'], label='calling', color=[0, 0, 255])

    cv2.imshow("res", annotator.result())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

