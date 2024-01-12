import contextlib
import time

import cv2
import numpy as np
import torch
from numba import njit

"""
用来放公共的工具函数
"""


class Profile(contextlib.ContextDecorator):
    """
    with Profile() as p:
        ...do something...
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
def xywh2xyxy(x: np.ndarray):
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
