import os
import cv2
import numpy as np
import torch
import onnxruntime as ort
from typing import List, Tuple
from pprint import pprint

from .utils import preprocess, np_non_max_suppression, Profile
from schemas.person import Person, PersonPose

from settings import MODEL_DIR

input_size = (256, 192)


def _box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def _center_scale_to_box(center, scale):
    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = torch.tensor(img).float()
    if img.max() > 1:
        img /= 255
    return img


def tt_transform(src, bbox):
    xmin, ymin, xmax, ymax = bbox
    # 根据边界框计算中心点和缩放比例
    center, scale = _box_to_center_scale(xmin, ymin, xmax - xmin, ymax - ymin, float(input_size[1]) / input_size[0])
    # scale = scale * 1.0
    inp_h, inp_w = input_size

    # 根据中心点、缩放比例以及输入图像的尺寸，计算仿射变换矩阵
    trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
    # 进行仿射变换  src(1067, 1915, 3) -> img(256, 192, 3)
    img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
    # 反映射处理后的边界框
    bbox = _center_scale_to_box(center, scale)

    img = im_to_torch(img)
    # 对图像进行归一化处理
    img[0].add_(-0.406)
    img[1].add_(-0.457)
    img[2].add_(-0.480)
    return img, bbox


def get_inps(boxes, ori_img):
    inps = torch.zeros([len(boxes), 3, input_size[0], input_size[1]])
    bboxes, scores = [], []
    for i, b in enumerate(boxes):
        box = b[:4]
        x, y, xx, yy = box
        score = float(b[-1])
        if xx - x > 0 and yy - y > 0:
            o, box = tt_transform(ori_img, box)
            inps[i] = o
            bboxes.append(box)
            scores.append(score)
    return inps, bboxes, scores


def get_max_pred_cuda_batched(heatmaps):
    # e.i. (4, 17, 64, 48) -> (4, 17, 48), (4, 17, 48)
    v, i = torch.max(heatmaps, dim=2)
    # e.i. (4, 17, 47) -> (4, 17), (4, 17)
    maxvals, ii = torch.max(v, dim=2)
    # e.i. (4, 17) -> (4, 17, 1)
    iia = ii.unsqueeze(-1)
    # e.i. iw => (4, 17, 1)
    iw = torch.gather(i, 2, iia)
    # e.i. preds => (4, 17, 2)
    preds = torch.cat([iia, iw], dim=2)
    maxvals = maxvals.unsqueeze(-1)

    mask = maxvals > 0
    pred_mask = torch.cat([mask, mask], dim=2)
    preds *= pred_mask
    return preds, maxvals

def get_max_pred_np_batched(heatmaps):
    # e.i. (4, 17, 64, 48) -> (4, 17, 48), (4, 17, 48)
    v = np.max(heatmaps, axis=2)
    i = np.argmax(heatmaps, axis=2)
    # e.i. (4, 17, 47) -> (4, 17), (4, 17)
    maxvals = np.max(v, axis=2)
    ii = np.argmax(v, axis=2)
    # e.i. (4, 17) -> (4, 17, 1)
    iia = np.expand_dims(ii, axis=-1)
    # e.i. iw => (4, 17, 1)
    iw = np.take_along_axis(i, iia, axis=2)
    # e.i. preds => (4, 17, 2)
    preds = np.concatenate([iia, iw], axis=2)
    maxvals = np.expand_dims(maxvals, axis=-1)

    mask = maxvals > 0
    pred_mask = np.concatenate([mask, mask], axis=2)
    preds *= pred_mask
    return preds, maxvals


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, trans):
    target_coords = np.zeros(coords.shape)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords


def heatmap_to_coord_batched(hms_batched, bboxes):
    b, n_joints, hm_h, hm_w = hms_batched.shape
    batched_preds, batched_maxvals = get_max_pred_np_batched(hms_batched)
    # e.i. (4, 17, 2), (4, 17, 1)

    pose_coords = []
    pose_scores = []
    for bi in range(b):
        bbox = bboxes[bi]
        coords = batched_preds[bi]
        preds = np.zeros_like(coords)
        # transform bbox to scale
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        center = np.array([xmin + w * 0.5, ymin + h * 0.5])
        scale = np.array([w, h])
        trans = get_affine_transform(center, scale, 0, [hm_w, hm_h], inv=1)

        for i in range(coords.shape[0]):
            preds[i] = transform_preds(coords[i], trans)

        pose_coords.append(preds)
        s = batched_maxvals[bi]
        pose_scores.append(s)

    return pose_coords, pose_scores


def vis_frame(img, kps, kps_scores, format='coco'):
    kp_num = 17
    if kps[0].shape[0] > 0:
        kp_num = kps[0].shape[0]
    if kp_num == 17:
        if format == 'coco':
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]
        elif format == 'mpii':
            l_pair = [
                (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
                (13, 14), (14, 15), (3, 4), (4, 5),
                (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
            ]
        else:
            raise NotImplementedError
    elif kp_num == 136:
        raise NotImplementedError
    elif kp_num == 26:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),  # Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),  # Foot
        ]
    else:
        raise NotImplementedError
    # c = (255, 158, 23)
    # c = (255, 0, 255)
    c = (0, 255, 0)
    # c = (255, 255, 0)
    for i in range(len(kps)):
        part_line = {}
        kp_preds = kps[i]
        kp_scores = kps_scores[i]
        if kp_num == 17:
            # kp_preds = torch.cat((kp_preds, torch.unsqueeze(
            #     (kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
            kp_preds = np.vstack([kp_preds, (kp_preds[5, :] + kp_preds[6, :]) / 2])
            # kp_scores = torch.cat((kp_scores, torch.unsqueeze(
            #     (kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
            kp_scores = np.vstack(
                [kp_scores, (kp_scores[5, :] + kp_scores[6, :]) / 2])

        # Draw keypoints
        vis_thres = 0.05 if kp_num == 136 else 0.4
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= vis_thres:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x), int(cor_y))
            cv2.circle(img, (int(cor_x), int(cor_y)),
                       1, c, 4, cv2.LINE_AA)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, c, 1, cv2.LINE_AA)
    return img


def vis_hm(ori_img, boxes, scores, hm):
    eval_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    if hm.shape[1] == 136:
        eval_joints = [*range(0, 136)]
    elif hm.shape[1] == 26:
        eval_joints = [*range(0, 26)]
    if len(boxes) == hm.shape[0]:
        pose_coords, pose_scores = heatmap_to_coord_batched(hm, boxes)
        ori_img = vis_frame(ori_img, pose_coords, pose_scores)
        return ori_img
    else:
        print('boxes and hm num not same, {} vs {}'.format(
            len(boxes), hm.shape[0]))


class PoseDetect:
    def __init__(self,
                 weight1=os.path.join(MODEL_DIR, 'yolov5x.onnx'),
                 weight2=os.path.join(MODEL_DIR, 'fast_res50_256x192_dynamic_simplified.onnx'),
                 is_gpu=False):
        self.weight_1 = weight1
        self.weight_2 = weight2
        self.conf_thres = 0.5
        self.iou_thres = 0.5
        self.device = torch.device('cuda' if torch.cuda.is_available() and is_gpu else 'cpu')
        self.providers = ['CPUExecutionProvider', 'CUDAExecutionProvider'] if is_gpu else ['CPUExecutionProvider']

        # person onnx
        self.session_1 = ort.InferenceSession(self.weight_1, providers=self.providers)
        self.output_names_1 = [x.name for x in self.session_1.get_outputs()]
        self.meta_1 = self.session_1.get_modelmeta().custom_metadata_map
        if 'stride' in self.meta_1:
            self.stride_1, self.names_1 = int(self.meta_1['stride']), eval(self.meta_1["names"])

        # fast-pose onnx
        self.pose_batch = 4
        self.session_2 = ort.InferenceSession(self.weight_2, providers=self.providers)
        self.output_names_2 = [i.name for i in self.session_2.get_outputs()]
        self.ip_h_2, self.ip_w_2 = self.session_2.get_inputs()[0].shape[2:]

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

    def pose_estimate(self, person_res, im0):
        # Pose preprocess
        inps, bboxes, scores = get_inps(person_res, im0)

        # Pose Estimation
        inps = inps.cpu().numpy()
        datalen = inps.shape[0]
        leftover = 0
        if (datalen) % self.pose_batch:
            leftover = 1
        num_batches = datalen // self.pose_batch + leftover
        hm = []
        for j in range(num_batches):
            inps_j = inps[j * self.pose_batch:min((j + 1) * self.pose_batch, datalen)]
            hm_j = self.session_2.run(self.output_names_2, {self.session_2.get_inputs()[0].name: inps_j})[0]
            hm.append(hm_j)
        hm = np.concatenate(hm)
        pose_coords, pose_scores = heatmap_to_coord_batched(hm, bboxes)
        return pose_coords, pose_scores

    def LQ_detect(self, im0):
        dt = [Profile()] * 3

        """person detect"""
        with dt[0]:
            person_res = self.human_detect(im0)
            print(len(person_res))
            print(person_res.shape)

        persons_res_list = []
        if person_res.shape[0] == 0:
            return None
        else:
            """pose estimate"""
            with dt[1]:
                pose_coords, pose_scores = self.pose_estimate(person_res, im0)
                print(len(pose_coords), len(pose_scores))

        for i, (person, pose_coord, pose_score) in enumerate(zip(person_res, pose_coords, pose_scores)):
            print(i, person.shape, pose_coord.shape, type(pose_coord), pose_score.shape, type(pose_score))
            person_box = [int(x) for x in person[:4]]
            person_res_obj = {
                'person_id': i,
                'person_box': person_box,
                'person_score': float(person[4]),
                'person_behaviors': {'smoking': [], 'calling': []},
                'person_pose': {'pose_coords': pose_coord.tolist(), 'pose_scores': [x[0] for x in pose_score]}
            }
            persons_res_list.append(person_res_obj)
        return persons_res_list


class PoseDetect_V2:
    def __init__(self, weight2=os.path.join(MODEL_DIR, 'fast_res50_256x192_dynamic_simplified.onnx'), is_gpu=False):
        self.weight2 = weight2
        self.device = torch.device('cuda' if torch.cuda.is_available() and is_gpu else 'cpu')
        self.providers = ['CPUExecutionProvider', 'CUDAExecutionProvider'] if is_gpu else ['CPUExecutionProvider']

        # load fast_pose onnx-model
        self.pose_batch = 4
        self.session_2 = ort.InferenceSession(self.weight2, providers=self.providers)
        self.output_names_2 = [i.name for i in self.session_2.get_outputs()]
        self.ip_h_2, self.ip_w_2 = self.session_2.get_inputs()[0].shape[2:]
        self.model_name = 'fast_pose'

    def warmup(self):
        im2 = np.random.randn(4, 3, self.ip_h_2, self.ip_w_2).astype(np.float32)
        for _ in range(2):
            wy = self.session_2.run(self.output_names_2, {self.session_2.get_inputs()[0].name: im2})
        return wy[0].shape

    def forward(self, person_res: np.ndarray, im0: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Pose Preprocess"""
        inps, bboxes, scores = get_inps(person_res, im0)

        """Pose Inference"""
        inps = inps.cpu().numpy()
        datalen = inps.shape[0]
        leftover = 0
        if (datalen) % self.pose_batch:
            leftover = 1
        num_batches = datalen // self.pose_batch + leftover
        hm = []
        for j in range(num_batches):
            inps_j = inps[j * self.pose_batch:min((j + 1) * self.pose_batch, datalen)]
            hm_j = self.session_2.run(self.output_names_2, {self.session_2.get_inputs()[0].name: inps_j})[0]
            hm.append(hm_j)
        hm = np.concatenate(hm)

        """Pose Postprocess"""
        pose_coords, pose_scores = heatmap_to_coord_batched(hm, bboxes)
        return pose_coords, pose_scores

    def record_result(self, detects_res: tuple, persons_res: List[Person]) -> List[Person]:
        model_name = self.model_name
        pose_coords, pose_scores = detects_res
        for pose_coord, pose_score, person_res in zip(pose_coords, pose_scores, persons_res):
            person_pose = PersonPose()
            person_pose.pose_coords = pose_coord.tolist()
            person_pose.pose_scores = [float(x[0]) for x in pose_score]
            person_res.person_poses = person_pose

        return persons_res


if __name__ == '__main__':
    im = cv2.imread(r'D:\ZHIHUI\Full_stack\Fastapi_Vue\main'
                    r'\backend\app\TaskRecord\8fe60e15-ebc3-456d-9a98-d305a2b839da'
                    r'\2023-10-27 13-37-35.jpg')
    pose_detect = PoseDetect()
    a, b = pose_detect.warmup()
    print(a, b)

    print("------------------------")
    c = pose_detect.LQ_detect(im)
    pprint(c)

