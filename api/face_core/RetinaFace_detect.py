import time
import cv2
import onnxruntime as ort
import numpy as np
from settings import MODEL_DIR
import os


def distance2bbox(anchor_centers, bboxes, max_shape=None):
    """
    Decode distance prediction to bounding box.
    ----------
        points: Shape=(n, 2) ===> n*[x, y]
        distance: bboxes (left, top, right, bottom)
        max_shape: shape of the image
    -------
    Returns
        Decoded bboxes
    """
    x1 = anchor_centers[:, 0] - bboxes[:, 0]
    y1 = anchor_centers[:, 1] - bboxes[:, 1]
    x2 = anchor_centers[:, 0] + bboxes[:, 2]
    y2 = anchor_centers[:, 1] + bboxes[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class RetinaFace:
    def __init__(self, model_file=os.path.join(MODEL_DIR, "det_10g.onnx"), input_size=(640, 640)):
        self.model_file = model_file
        session_options = ort.SessionOptions()
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_file, providers=providers, session_options=session_options)

        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars(input_size)

    def _init_vars(self, input_size):
        input_metadata = self.session.get_inputs()[0]
        input_shape = input_metadata.shape
        if isinstance(input_shape[2], str):
            self.input_size = input_size
        else:
            self.input_shape = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        self.input_name = input_metadata.name
        self.input_mean = 127.5
        self.input_std = 128.0

        outputs = self.session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        # print(output_names)
        self.output_names = output_names
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1

        if len(output_names) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True

    def prepare(self, img):
        input_size = self.input_size
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        return det_img, det_scale

    def forward(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []

        # input_size   (h, w) ----> (w, h)
        input_size = tuple(img.shape[0:2][::-1])
        # blob -> (1, 3, 640, 640), size参数为（w, h)
        blob = cv2.dnn.blobFromImage(img, 1.0 / self.input_std, input_size,
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        # fmc = 3
        fmc = self.fmc

        for index, stride in enumerate(self._feat_stride_fpn):
            # scores: 448, 471, 494 ----> [12800, 1], [3200, 1], [800, 1]
            scores = net_outs[index]

            # bbox: 451, 474, 497 ----> [12800, 4], [3200, 4], [800, 4]
            bboxes = net_outs[index + fmc] * stride

            # kps: 454, 477, 500 ----> [12800, 10], [3200, 10], [800, 10]
            if self.use_kps:
                key_points = net_outs[index + 6] * stride

            # 640//[32, 16, 8] ==> (80, 80), (40, 40), (20, 20)
            height = input_height // stride
            width = input_width // stride
            k = height * width
            # (80, 80, 8), (40, 40, 16), (20, 20, 32)
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # ax = np.arange(width)
                # ay = np.arange(height)
                # xv, yv = np.meshgrid(ax, ay)
                # anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)
                # print(anchor_centers)

                # (80, 80, 2), (40, 40, 2), (20, 20, 2)
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                # print(anchor_centers.shape)

                # (6400, 2), (1600, 2), (400, 2)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    # anchor_centers: (12800, 2), (3200, 2), (800, 2)
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            # select index which > 0.5
            selected_indexes = np.where(scores >= threshold)[0]
            selected_scores = scores[selected_indexes]
            scores_list.append(selected_scores)

            selected_bboxes = distance2bbox(anchor_centers, bboxes)
            # print(selected_bboxes.shape)
            selected_bboxes = selected_bboxes[selected_indexes]
            bboxes_list.append(selected_bboxes)

            if self.use_kps:
                # key points pred
                kpss = distance2kps(anchor_centers, key_points)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                selected_key_points = kpss[selected_indexes]
                kpss_list.append(selected_key_points)
            # print(selected_bboxes.shape)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size=None, max_num=0, metric='default'):
        det_img, det_scale = self.prepare(img)
        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                              det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


if __name__ == "__main__":

    s = time.time()
    D = RetinaFace()
    print(time.time()-s)
