import paddle
import argparse
import cv2
import numpy as np
import sys
import os
import logging
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from models.arcface import IRBlock, ResNet
from models.model import FaceSwap, l2_norm
from utils.align_face import align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel

# attacks.py가 있는 폴더 경로 추가 (예시: /home/user/project/utils/)
attacks_dir = "/home/kjh/dev/capstone/disrupting-deepfakes/stargan/"  # 🔹 여기에 실제 `attacks.py`가 있는 폴더 경로를 입력
sys.path.append(attacks_dir)

# 이제 import 가능
from attacks import LinfPGDAttack


# 로그 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def apply_pgd_attack(source_img_path, output_dir, model, use_gpu, epsilon=0.03, alpha=0.005, steps=10):
    """ResNet 모델을 사용하여 PGD 공격을 수행"""
    try:
        os.makedirs(output_dir, exist_ok=True)

        device = torch.device("cuda" if use_gpu else "cpu")

        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(source_img_path).convert('RGB')
        X_nat = transform(image).unsqueeze(0).to(device)  # ✅ PyTorch Tensor로 변환 후 GPU로 이동

        # PGD 공격을 수행하는 LinfPGDAttack 객체 생성
        pgd_attacker = LinfPGDAttack(model=model, device=device, epsilon=epsilon, k=steps, a=alpha)

        # PGD 공격 수행
        logging.info("⚡ PGD 공격 시작...")
        x_adv, perturbation = pgd_attacker.perturb(X_nat, y=None)

        if x_adv is None:
            logging.error("❌ PGD 공격 실패! 딥페이크 수행을 중단합니다.")
            return None, None

        # ✅ GPU → CPU 변환 추가
        X_nat_cpu = X_nat.cpu()  
        x_adv_cpu = x_adv.cpu()

        # PGD 공격 전후 이미지 저장
        before_pgd_path = os.path.join(output_dir, "source_before_pgd.jpg")
        after_pgd_path = os.path.join(output_dir, "source_after_pgd.jpg")
        save_image(X_nat_cpu, before_pgd_path)  # ✅ GPU 텐서를 CPU로 변환 후 저장
        save_image(x_adv_cpu, after_pgd_path)   # ✅ GPU 텐서를 CPU로 변환 후 저장

        logging.info(f"✅ PGD 공격 후 source 저장됨: {after_pgd_path}")

        return before_pgd_path, after_pgd_path
    except Exception as e:
        logging.error(f"PGD 공격 실패: {e}")
        return None, None  # PGD 실패 시 이미지 없음


def perform_deepfake(source_img_path, target_img_path, output_dir, use_gpu, mode):
    """image_test_origin.py를 호출하여 딥페이크 수행"""
    if source_img_path is None:
        logging.warning(f"❌ {mode} 파일이 존재하지 않아 딥페이크 수행을 건너뜁니다.")
        return

    try:
        # ✅ 원본 source → origin, PGD 공격된 source → PGDattack
        if mode == "origin":
            output_filename = "origin"
        else:
            output_filename = "PGDattack"

        output_dir_mode = os.path.join(output_dir, output_filename)
        os.makedirs(output_dir_mode, exist_ok=True)

        cmd = f"python image_test_origin.py --source_img_path {source_img_path} " \
              f"--target_img_path {target_img_path} --output_dir {output_dir_mode} --use_gpu {use_gpu}"
        logging.info(f"🚀 딥페이크 실행: {cmd}")
        os.system(cmd)
    except Exception as e:
        logging.error(f"💥 딥페이크 수행 중 오류 발생: {e}")


def main(args):
    """PGD 공격을 수행한 후, 두 가지 딥페이크 수행"""
    device = "gpu" if args.use_gpu else "cpu"

    paddle.set_device(device)

    # 기존 ResNet 모델 로드
    logging.info("🔵 ResNet 모델 로드 중...")
    id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])

    if id_net is None:
        logging.error("💥 ResNet 모델 생성 실패!")
        return

    model_path = './checkpoints/arcface.pdparams'
    if not os.path.exists(model_path):
        logging.error(f"💥 모델 파라미터 파일이 존재하지 않음: {model_path}")
        return
    
    try:
        model_state_dict = paddle.load(model_path)
        if model_state_dict is None:
            logging.error(f"💥 paddle.load() 실패: {model_path}")
            return

        id_net.set_dict(model_state_dict)
        id_net.eval()

        logging.info("✅ ResNet 모델 로드 완료!")
    except Exception as e:
        logging.error(f"💥 모델 로딩 중 오류 발생: {e}")
        return

    # 🔥 PGD 공격 수행
    logging.info("🔥 PGD 공격을 수행 중...")
    before_pgd_path, after_pgd_path = apply_pgd_attack(args.source_img_path, args.output_dir, id_net, args.use_gpu)

    # **PGD 공격 실패 시 원본 딥페이크만 실행**
    if before_pgd_path is None:
        logging.warning("❌ PGD 공격이 실패하여 PGDattack 딥페이크를 건너뜁니다.")
        return  # PGD 공격 실패 시 프로그램 종료

    # 1️⃣ **원본 source로 딥페이크 수행 (origin)**
    logging.info("🔵 원본 source를 사용한 딥페이크 수행 중...")
    perform_deepfake(before_pgd_path, args.target_img_path, args.output_dir, args.use_gpu, mode="origin")

    # 2️⃣ **PGD 적용된 source로 딥페이크 수행 (PGDattack)**
    if after_pgd_path:
        logging.info("🟠 PGD 공격된 source를 사용한 딥페이크 수행 중...")
        perform_deepfake(after_pgd_path, args.target_img_path, args.output_dir, args.use_gpu, mode="PGDattack")

    logging.info("✅ 모든 딥페이크 테스트 완료!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")
    parser.add_argument('--source_img_path', type=str, help='Path to the source image')
    parser.add_argument('--target_img_path', type=str, help='Path to the target images')
    parser.add_argument('--output_dir', type=str, default='results', help='Path to the output directory')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--merge_result', type=bool, default=True)
    parser.add_argument('--need_align', type=bool, default=True)
    parser.add_argument('--use_gpu', type=bool, default=False)

    args = parser.parse_args()
    main(args)
