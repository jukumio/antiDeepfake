import torch
import numpy as np
import argparse
import os
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F

from color_space import *
from data_loader import get_loader
from utils import *
from model import Generator, Discriminator

from skimage.metrics import structural_similarity as ssim_func  
from skimage.transform import resize

def main():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')

    # Data configuration.
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--attack_iters', type=int, default=100, help='Number of iterations for Lab Attack')
    parser.add_argument('--resume_iters', type=int, default=200000, help='resume training from this step')
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    parser.add_argument('--selected_attrs', '--list', nargs='+', default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    parser.add_argument('--celeba_image_dir', type=str, default='/home/kjh/dev/capstone/disrupting-deepfakes/stargan/data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='/home/kjh/dev/capstone/disrupting-deepfakes/stargan/data/celeba/list_attr_celeba.txt')
    parser.add_argument('--model_save_dir', type=str, default='/home/kjh/dev/capstone/disrupting-deepfakes/stargan/stargan_celeba_256/models')
    parser.add_argument('--result_dir', type=str, default='/home/kjh/dev/capstone/AntiForgery/results')

    config = parser.parse_args()
    os.makedirs(config.result_dir, exist_ok=True)

    celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                               config.celeba_crop_size, config.image_size, config.batch_size,
                               'CelebA', 'test', 0)

    # Load StarGAN models
    G = Generator(config.g_conv_dim, config.c_dim, config.g_repeat_num).cuda()
    D = Discriminator(config.image_size, config.d_conv_dim, config.c_dim, config.d_repeat_num).cuda()

    # Model weights selection logic
    model_iters = config.test_iters if config.test_iters else config.resume_iters
    print('Loading the trained models from step {}...'.format(model_iters))

    G_path = os.path.join(config.model_save_dir, '{}-G.ckpt'.format(model_iters))
    D_path = os.path.join(config.model_save_dir, '{}-D.ckpt'.format(model_iters))
    load_model_weights(G, G_path)
    D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
    print("Model loading successful")

    for i, (x_real, c_org) in enumerate(celeba_loader):
        x_real = x_real.cuda()
        c_trg_list = create_labels(c_org, config.c_dim, 'CelebA', config.selected_attrs)

        # Generate adversarial example
        x_adv, _ = lab_attack(x_real, c_trg_list, G, iter=config.attack_iters)

        # Save original image
        save_image(denorm(x_real.data.cpu()), os.path.join(config.result_dir, f'{i}_original.jpg'))

        for idx, c_trg in enumerate(c_trg_list):
            print(f'Processing image {i}, class {idx}')

            with torch.no_grad():
                gen_noattack, _ = G(x_real, c_trg)  # No-attack (원본 사용)
                gen, _ = G(x_adv, c_trg)  # Attack (perturbed 이미지 사용)

            # 개별 이미지 저장
            save_image(denorm(gen_noattack.data.cpu()), os.path.join(config.result_dir, f'{i}_class{idx}_noattack.jpg'))
            save_image(denorm(gen.data.cpu()), os.path.join(config.result_dir, f'{i}_class{idx}_adv.jpg'))

        if i == 50:  # Stop after 50 images
            break

    print("Image saving complete.")

if __name__ == '__main__':
    main()
