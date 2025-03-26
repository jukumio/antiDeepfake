import torch
import numpy as np
import argparse
import os
import sys
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_func  
from skimage.transform import resize

from color_space import *
from data_loader import get_loader
from utils import *
from model import Generator, Discriminator

attacks_dir = "/home/kjh/dev/capstone/disrupting-deepfakes/stargan/"  # 🔹 여기에 실제 `attacks.py`가 있는 폴더 경로를 입력
sys.path.append(attacks_dir)
# 공격 모듈 추가
from attacks import LinfPGDAttack  

def main():
    parser = argparse.ArgumentParser()

    # 모델 설정
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')

    # 데이터 설정
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--attack_iters', type=int, default=100, help='Number of iterations for Lab Attack')
    parser.add_argument('--resume_iters', type=int, default=200000, help='resume training from this step')
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    parser.add_argument('--selected_attrs', '--list', nargs='+', default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    parser.add_argument('--celeba_image_dir', type=str, default='/home/kjh/dev/capstone/disrupting-deepfakes/stargan/data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='/home/kjh/dev/capstone/disrupting-deepfakes/stargan/data/celeba/list_attr_celeba.txt')
    parser.add_argument('--model_save_dir', type=str, default='/home/kjh/dev/capstone/disrupting-deepfakes/stargan/stargan_celeba_256/models/')
    parser.add_argument('--result_dir', type=str, default='/home/kjh/dev/capstone/AntiForgery/results')

    config = parser.parse_args()
    os.makedirs(config.result_dir, exist_ok=True)

    celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                               config.celeba_crop_size, config.image_size, config.batch_size,
                               'CelebA', 'test', 0)

    # StarGAN 모델 불러오기
    G = Generator(config.g_conv_dim, config.c_dim, config.g_repeat_num).cuda()
    D = Discriminator(config.image_size, config.d_conv_dim, config.c_dim, config.d_repeat_num).cuda()

    model_iters = config.test_iters if config.test_iters else config.resume_iters
    print('Loading the trained models from step {}...'.format(model_iters))

    G_path = os.path.join(config.model_save_dir, '{}-G.ckpt'.format(model_iters))
    D_path = os.path.join(config.model_save_dir, '{}-D.ckpt'.format(model_iters))
    load_model_weights(G, G_path)
    D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
    print("Model loading successful")

    # 🔹 공격 객체 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack = LinfPGDAttack(model=G, device=device, epsilon=0.05, k=10, a=0.01)

    # 🔹 이미지 변환 및 적대적 공격 적용
    for i, (x_real, c_org) in enumerate(celeba_loader):
        x_real = x_real.cuda()
        c_trg_list = create_labels(c_org, config.c_dim, 'CelebA', config.selected_attrs)

        # ✅ PGD 공격 적용 시 도메인 레이블 추가
        # PGD 공격 적용 (도메인 레이블 `c_trg_list[0]` 추가)
        x_adv, _ = attack.perturb(x_real, x_real, c_trg_list[0])


        # 원본 이미지 저장
        save_image(denorm(x_real.data.cpu()), os.path.join(config.result_dir, f'{i}_original.jpg'))
        save_image(denorm(x_adv.data.cpu()), os.path.join(config.result_dir, f'{i}_adv.jpg'))  # 공격 적용된 이미지 저장

        for idx, c_trg in enumerate(c_trg_list):
            print(f'Processing image {i}, class {idx}')

            with torch.no_grad():
                gen_noattack, _ = G(x_real, c_trg)  # 공격 없이 생성된 이미지
                gen, _ = G(x_adv, c_trg)  # 공격된 이미지를 입력으로 한 생성된 이미지

            # 생성된 이미지 저장
            save_image(denorm(gen_noattack.data.cpu()), os.path.join(config.result_dir, f'{i}_class{idx}_noattack.jpg'))
            save_image(denorm(gen.data.cpu()), os.path.join(config.result_dir, f'{i}_class{idx}_adv.jpg'))

        if i == 50:  # 50개의 이미지 처리 후 종료
            break


    print("Image saving complete.")

if __name__ == '__main__':
    main()
