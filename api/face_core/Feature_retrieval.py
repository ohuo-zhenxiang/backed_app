import pickle
import os
import cv2
from annoy import AnnoyIndex

from api.face_core.ArcFace_extract import ArcFaceOrt
from settings import FEATURE_LIB
from pathlib import Path


class AnnoyTree:
    def __init__(self):

        self.index_tree = AnnoyIndex(512, metric='dot')
        self.names_list = []
        self.ids_list = []

    def create_tree(self, vector_list: list, save_filename: str):
        for i, (fid, name, feature) in enumerate(vector_list):
            feature = pickle.loads(feature)
            if len(feature) == 512:
                self.index_tree.add_item(i, feature)
                self.ids_list.append(fid)
                self.names_list.append(name)
            else:
                return False, fid, name

        self.index_tree.build(512, n_jobs=-1)

        self.index_tree.save(os.path.join(FEATURE_LIB, f"{save_filename}.ann"))
        with open(os.path.join(FEATURE_LIB, f"{save_filename}.pickle"), "wb") as f:
            pickle.dump((self.ids_list, self.names_list), f)

        return True, None, None


class Retrieval:
    recognizer = ArcFaceOrt()

    def __init__(self, task_id):
        # s = time.time()
        with open(os.path.join(FEATURE_LIB, f"{task_id}.pickle"), "rb") as f:
            self.fids, self.names = pickle.load(f)

        self.index = AnnoyIndex(512, metric='dot')
        self.index.load(os.path.join(FEATURE_LIB, f"{task_id}.ann"))

        # end = round(time.time() - s, 2)
        # logger.success(f"Retrieval init..., take times: {end}s")

    def search(self, query_feature, k=10):
        indices, distances = self.index.get_nns_by_vector(query_feature, k, search_k=10 ** 9, include_distances=True)
        return indices, distances

    def run_retrieval(self, image, key_points):
        query_feature_raw = self.recognizer.feature_extract(image, key_points=key_points)

        query_feature = query_feature_raw.copy().tolist()[0]
        indices, distances = self.search(query_feature, k=20)

        result_names = [self.names[i] for i in indices]

        # print(f"result_names: {result_names[:5]}")
        # print(f"result_distances: {distances[:5]}")
        #
        # print("-----------------------------------")
        return indices[0], distances[0]

    @staticmethod
    def draw_result(faces_reg_name, query_face_location, raw_img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y, w, h = tuple(query_face_location)
        cv2.rectangle(raw_img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)
        cv2.putText(raw_img, str(faces_reg_name), (x + 5, y - 5), font, 0.8, (0, 255, 0), 2)
        return raw_img


if __name__ == "__main__":
    pass
