import glob
import os
import sys

from pathlib import Path

import cv2
import torch
import numpy as np
import onnxruntime as ort
from loguru import logger
from settings import MODEL_DIR


def sphincter(src_img, bbox, output_size=(80, 80), crop=True, scale_param=1.7):
    """
    :param scale_param: 括的系数
    :param src_img: np.ndarray
    :param bbox: xyxy - (x1, y1), (x2, y2)
    :param output_size: 80x80
    :param crop: 要不要裁剪，b废话
    :return: croped-image
    """
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1
    src_h, src_w, _ = np.shape(src_img)

    scale = min((src_h-1)/box_h, (src_w-1)/box_w, scale_param)

    center_x, center_y = box_w/2+x1, box_h/2+y1
    new_width, new_height = box_w * scale, box_h * scale

    left_top_x = max(0, center_x - new_width/2)
    left_top_y = max(0, center_y - new_height/2)
    right_bottom_x = min(src_w-1, center_x + new_width/2)
    right_bottom_y = min(src_h-1, center_y + new_height/2)

    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def sphincter_crop(src_img, bbox, output_size=(80, 80), crop=True, scale_param=1.7):
    x1, y1, x2, y2 = sphincter(src_img, bbox, output_size, crop, scale_param)
    img = src_img[y1: y2+1, x1: x2+1]
    dst_img = cv2.resize(img, (output_size[0], output_size[1]))
    return dst_img, (x1, y1, x2, y2)


class SilentFaceAntiSpoofing:
    def __init__(self, model_file=os.path.join(MODEL_DIR, '2.7_80x80_MiniFASNetV2.onnx'), gpu=False):
        self.model_file = model_file
        self.is_gpu = gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.is_gpu else 'cpu')
        self.providers = ['CPUExecutionProvider', 'CUDAExecutionProvider'] if self.is_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_file, providers=self.providers)

        self.input_name = self.session.get_inputs()[0].name
        self.input_size = self.session.get_inputs()[0].shape[2:]
        self.output_name = self.session.get_outputs()[0].name
        self.output_size = self.session.get_outputs()[0].shape

    def warmup(self, imgsz=(1, 3, 80, 80)):
        input_image = np.random.randn(*imgsz).astype(np.float32)
        for _ in range(2):
            wy = self.session.run([self.output_name], {self.input_name: input_image})[0]
            wy = self.softmax(wy)
        return wy

    @staticmethod
    def softmax(prob):
        total = np.sum(np.exp(prob - np.max(prob)))
        result = np.exp(prob - np.max(prob)) / total
        return result

    def forward(self, box, img):
        crop_img, crop_box = sphincter_crop(img, box)

        crop_img = np.transpose(crop_img, (2, 0, 1))
        crop_img = np.expand_dims(crop_img, axis=0)
        crop_img = crop_img.astype(np.float32)
        prediction = self.session.run([self.output_name], {self.input_name: crop_img})[0]
        result = self.softmax(prediction)
        label = np.argmax(result)
        trueness = result[0][label]

        return label, trueness


if __name__ == '__main__':
    img = cv2.imread('../../test/test.jpg')
    sfas = SilentFaceAntiSpoofing()
    res_label, res_trueness = sfas.forward(box=(0, 0, 295, 413), img=img)
    print(res_label, res_trueness)



