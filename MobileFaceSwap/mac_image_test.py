import os
import argparse
import cv2
import numpy as np
import paddle
import sys
sys.path.append('../capstone/MobileFaceSwap')

from utils.prepare_data_o import LandmarkModel
from utils.align_face import back_matrix, dealign, align_img
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.util import cv2paddle, paddle2cv

def get_id_emb(id_net, id_img_path):
    id_img = cv2.imread(id_img_path)
    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2paddle(id_img)
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std
    
    id_emb, id_feature = id_net(id_img)
    id_emb = l2_norm(id_emb)
    return id_emb, id_feature

def image_test(args):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
    
    print("🚀 모델 로드 시작")
    landmarkModel = LandmarkModel(name=os.path.join(CHECKPOINTS_DIR, 'landmarks'))
    landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640))
    
    faceswap_model = FaceSwap(False)
    
    id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
    id_net.set_dict(paddle.load(os.path.join(CHECKPOINTS_DIR, 'arcface.pdparams')))
    id_net.eval()
    
    weight = paddle.load(os.path.join(CHECKPOINTS_DIR, 'MobileFaceSwap_224.pdparams'))
    base_path = args.source_img_path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
    id_emb, id_feature = get_id_emb(id_net, base_path + '_aligned.png')
    
    faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
    faceswap_model.eval()
    
    if os.path.isfile(args.target_img_path):
        img_list = [args.target_img_path]
    else:
        img_list = [os.path.join(args.target_img_path, x) for x in os.listdir(args.target_img_path) if x.endswith(('png', 'jpg', 'jpeg'))]
    
    for img_path in img_list:
        origin_att_img = cv2.imread(img_path)
        base_path = img_path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        att_img = cv2.imread(base_path + '_aligned.png')
        
        if att_img is None:
            raise ValueError(f"❌ 얼굴 정렬된 이미지가 로드되지 않음: {base_path + '_aligned.png'}")

        att_img = cv2paddle(att_img)
        
        print("🚀 얼굴 변환 시작")
        try:
            res, mask = faceswap_model(att_img)
        except Exception as e:
            raise RuntimeError(f"❌ faceswap_model 실패: {str(e)}")

        res = paddle2cv(res)

        if res is None or res.shape[0] == 0:
            raise ValueError("❌ 변환된 이미지가 None 또는 비어 있음!")


        if args.merge_result:
            back_matrix_data = np.load(base_path + '_back.npy')
            mask = np.transpose(mask[0].numpy(), (1, 2, 0))
            res = dealign(res, origin_att_img, back_matrix_data, mask)
        
        output_file = os.path.join(args.output_dir, os.path.basename(img_path))
        success = cv2.imwrite(output_file, res)
        if not success:
            print(f"❌ 이미지 저장 실패: {output_file}")

        print(f"✅ 변환 완료: {output_file}")


def face_align(landmarkModel, image_path, merge_result=False, image_size=224):
    if os.path.isfile(image_path):
        img_list = [image_path]
    else:
        img_list = [os.path.join(image_path, x) for x in os.listdir(image_path) if x.endswith(('png', 'jpg', 'jpeg'))]
    
    for path in img_list:
        img = cv2.imread(path)
        landmark = landmarkModel.get(img)
        if landmark is not None:
            base_path = path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
            aligned_img, back_matrix_data = align_img(img, landmark, image_size)
            cv2.imwrite(base_path + '_aligned.png', aligned_img)
            if merge_result:
                np.save(base_path + '_back.npy', back_matrix_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_img_path", type=str, required=True)
    parser.add_argument("--target_img_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--merge_result", type=bool, default=True)
    parser.add_argument("--need_align", type=bool, default=True)
    
    args = parser.parse_args()
    
    if args.need_align:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')

        landmarkModel = LandmarkModel(name=os.path.join(CHECKPOINTS_DIR, 'landmarks'))
        landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640))
        face_align(landmarkModel, args.source_img_path)
        face_align(landmarkModel, args.target_img_path, args.merge_result, args.image_size)
    
    os.makedirs(args.output_dir, exist_ok=True)
    image_test(args)