import onnxruntime as ort
import cv2
import time
import os

from api.face_core import FaceAlign
from settings import MODEL_DIR


class ArcFaceOrt:
    def __init__(self, model_path=os.path.join(MODEL_DIR, 'w600k_r50.onnx'), gpu=False):
        self.model_path = model_path
        # self.providers = ['CUDAExecutionProvider'] if gpu else ['CPUExecutionProvider']
        self.providers = ['CPUExecutionProvider']

        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = 4
        session_options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(model_path, providers=self.providers, session_options=session_options)

        input_a = self.session.get_inputs()[0]
        self.input_size = tuple(input_a.shape[2:4][::-1])
        self.input_name = input_a.name

        output_a = self.session.get_outputs()[0]
        self.output_name = output_a.name
        self.output_shape = output_a.shape

        self.input_mean = 127.5
        self.input_std = 127.5

    def get_features(self, face_img):
        blob = cv2.dnn.blobFromImage(face_img, 1.0 / self.input_std, self.input_size,
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True, crop=False)
        net_out = self.session.run(None, {self.input_name: blob})[0]
        return net_out

    def feature_extract(self, img, key_points):
        face_img = FaceAlign.norm_crop(img, landmark=key_points, image_size=self.input_size[0])
        embedding = self.get_features(face_img)
        return embedding


def main_run():
    from RetinaFace_detect import RetinaFace

    detector = RetinaFace()
    recognizer = ArcFaceOrt()

    s = time.time()
    img = cv2.imread('./data/LYF/1.jpg')
    cimg = img.copy()
    bboxes_5, kpss = detector.detect(cimg)
    box = bboxes_5[0][:-1].astype(int)
    detect_score = round(bboxes_5[0][-1], 2)
    kps = kpss[0]
    cv2.rectangle(cimg, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=2)
    cv2.putText(cimg, f"size: {(box[2] - box[0], box[3] - box[1])}", (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0, 0, 255), thickness=2)

    embedding_features = recognizer.feature_extract(img, key_points=kps)
    print(embedding_features.shape, detect_score)
    print(round(time.time() - s, 2))
    cv2.imshow("result", cimg)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # cProfile.run("main_run()")
    a = ArcFaceOrt()
