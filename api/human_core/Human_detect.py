import contextlib
import os
import time
from typing import List

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision

from .utils import preprocess, np_non_max_suppression
from schemas.person import Person
from settings import MODEL_DIR


class Profile(contextlib.ContextDecorator):
    """
    with Profile() as p:
        ...
    take_times = p.dt
    """

    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            # GPU计算时没法算时间
            torch.cuda.synchronize()
        return time.time()


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


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top - left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


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


def non_max_suppression(
        prediction: torch.Tensor,  # e.i. Tensor(1, 25200, 85)
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlopping detections
    Returns:
        list of detections, on (n, 6) tensor per image [xyxy, conf, cls]
    """
    # check 两个阈值是否在合理范围内
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid Iou {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    # xc 返回的是是否大于thres的bool值

    # Settings
    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.5 * bs
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        # Apply constraints 上约束
        x = x[xc[xi]]  # confidence

        if not x.shape[0]:
            return None

        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        conf, j = x[:, 5:mi].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            return None
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break
    return output


class HumanDetect:
    def __init__(self, model_file=os.path.join(MODEL_DIR, "yolov5x.onnx"), gpu=False, conf_thres=0.35, iou_thres=0.45):
        self.model_file = model_file
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.is_gpu = gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.is_gpu else 'cpu')
        self.providers = ['CPUExecutionProvider', 'CUDAExecutionProvider'] if self.is_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_file, providers=self.providers)
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.meta = self.session.get_modelmeta().custom_metadata_map
        if 'stride' in self.meta:
            self.stride, self.names = int(self.meta['stride']), eval(self.meta['names'])

    def forward(self, im: torch.Tensor):
        # YOLOv5 inference
        b, ch, h, w = im.shape
        im = im.cpu().numpy()
        y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """Warmup model by running inference once"""
        if torch.cuda.is_available() and self.is_gpu:
            im = torch.empty(*imgsz, dtype=torch.float, device=torch.device('cuda'))
            for _ in range(1):
                self.forward(im)

    def human_detect(self, im0: np.ndarray):
        dt = (Profile(), Profile(), Profile())
        """preprocess"""
        with dt[0]:
            im = letterbox(im0)[0]  # resize and padded
            im = im.transpose((2, 0, 1))[::-1]  # HWC => CHW, BGR => RGB
            im = np.ascontiguousarray(im)  # 创建连续的副本
            im = torch.from_numpy(im).to(self.device)
            im = im.float()
            im /= 255  # 归一化
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        """Inference"""
        with dt[1]:
            pred = self.forward(im)

        """NMS"""
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres, classes=[0, ],
                                       max_det=1000)

        if pred is None:
            return None
        else:
            det = pred[0]
            if len(det):
                # print("front", det)
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                return det


class HumanDetect_V2:
    def __init__(self, weight1=os.path.join(MODEL_DIR, 'yolov5x.onnx'), is_gpu=False):
        self.weight_1 = weight1
        self.conf_thres = 0.5
        self.iou_thres = 0.5
        self.device = torch.device('cuda' if torch.cuda.is_available() and is_gpu else 'cpu')
        self.providers = ['CPUExecutionProvider', 'CUDAExecutionProvider'] if is_gpu else ['CPUExecutionProvider']

        # load person onnx-model
        self.session_1 = ort.InferenceSession(self.weight_1, providers=self.providers)
        self.output_names_1 = [x.name for x in self.session_1.get_outputs()]
        self.meta_1 = self.session_1.get_modelmeta().custom_metadata_map
        if 'stride' in self.meta_1:
            self.stride_1, self.names_1 = int(self.meta_1['stride']), eval(self.meta_1['names'])

    def warmup(self):
        im1 = np.ones((1, 3, 640, 640), dtype=np.float32)
        for _ in range(2):
            dz = self.session_1.run(self.output_names_1, {self.session_1.get_inputs()[0].name: im1})

    def forward(self, im0: np.ndarray):
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

    @staticmethod
    def record_result(res) -> List[Person]:
        persons_res_list = []
        for i, person in enumerate(res):
            person_res = Person()
            person_res.person_id = i
            person_res.person_box = [int(x) for x in person[:4]]
            person_res.person_score = float(person[4])
            persons_res_list.append(person_res)
        return persons_res_list
