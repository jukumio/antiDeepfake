import os
import cv2
import numpy as np
import glob
import os.path as osp
from insightface.model_zoo import model_zoo


class LandmarkModel():
    def __init__(self, name, root='./checkpoints'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)

        if not onnx_files:
            raise FileNotFoundError(f"❌ No ONNX model found in {osp.join(root, name)}")

        for onnx_file in onnx_files:
            if '_selfgen_' in onnx_file:
                continue
            model = model_zoo.get_model(onnx_file)
            taskname = model.taskname

            # 🔥 여기에서 scrfd → detection으로 변경
            if taskname == 'scrfd':  
                taskname = 'detection'

            print(f"✅ Model loaded: {onnx_file} - Task: {taskname}")

            if taskname not in self.models:
                self.models[taskname] = model
            else:
                print(f"⚠️ Duplicate model task ignored: {onnx_file} - Task: {taskname}")
                del model

        # 로드된 모델 목록 출력
        print(f"🔍 Loaded models: {list(self.models.keys())}")

        if 'detection' not in self.models:
            raise AssertionError(f"❌ 'detection' model not found! Loaded models: {list(self.models.keys())}")

        self.det_model = self.models['detection']



    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode ='None'):
        self.det_thresh = det_thresh
        self.mode = mode
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)


    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return None
        det_score = bboxes[..., 4]

        # select the face with the hightest detection score
        best_index = np.argmax(det_score)

        kps = None
        if kpss is not None:
            kps = kpss[best_index]
        return kps

    def gets(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric='default')
        return kpss