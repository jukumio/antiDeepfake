import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import PIL.Image
import argparse

from dnnlib import util as dnnlib_util

# LPIPS 네트워크 로딩 (vgg16 기반)
def load_lpips(device):
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib_util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)
    return vgg16

# 이미지 전처리
def preprocess_image(image_path, resolution, device):
    img = PIL.Image.open(image_path).convert('RGB')
    img = img.resize((resolution, resolution), PIL.Image.LANCZOS)
    img = np.array(img).transpose(2, 0, 1)  # CHW
    img = torch.tensor(img, dtype=torch.float32, device=device)
    return img

def main(args):
    device = torch.device('mps') if args.use_mps and torch.backends.mps.is_available() else torch.device('cpu')

    print(f"[INFO] Loading LPIPS model on device {device}...")
    lpips_model = load_lpips(device)

    print(f"[INFO] Loading target image: {args.target}")
    target_img = preprocess_image(args.target, resolution=256, device=device)
    target_img = target_img.unsqueeze(0)  # (1, C, H, W)
    target_features = lpips_model(target_img, resize_images=False, return_lpips=True)

    best_dist = float('inf')
    best_w = None
    w_list = []

    print(f"[INFO] Searching in directory: {args.w_candidates}")
    for root, dirs, files in os.walk(args.w_candidates):
        for file in files:
            if file.endswith('.npz'):
                path = os.path.join(root, file)
                data = np.load(path)
                if 'w' not in data:
                    continue
                w = torch.tensor(data['w'], device=device, dtype=torch.float32)
                w_list.append(w)

                # 이미지 생성은 하지 않고 LPIPS 특성으로 거리 계산
                from training.networks import Generator
                from legacy import load_network_pkl
                with dnnlib_util.open_url(args.network) as f:
                    net = load_network_pkl(f)['G_ema']
                init_kwargs = net.init_kwargs
                init_kwargs['synthesis_kwargs']['num_fp16_res'] = 0
                init_kwargs['synthesis_kwargs']['conv_clamp'] = None
                G = Generator(**init_kwargs).eval().requires_grad_(False).to(device).float()
                G.load_state_dict(net.state_dict())

                img = G.synthesis(w, noise_mode='const')
                img = (img + 1) * 127.5
                img = F.interpolate(img, size=(256, 256), mode='area')
                feat = lpips_model(img, resize_images=False, return_lpips=True)

                dist = (target_features - feat).square().sum()
                if dist < best_dist:
                    best_dist = dist
                    best_w = w

    if best_w is not None:
        print(f"[INFO] Saving closest w to {args.outpath}")
        np.savez(args.outpath, w=best_w.cpu().numpy())
    else:
        print("[ERROR] No valid .npz files found in the candidate directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Path to target image (e.g. JPG)")
    parser.add_argument("--w_candidates", required=True, help="Directory containing .npz files")
    parser.add_argument("--network", required=True, help="Path to original StyleGAN network .pkl")
    parser.add_argument("--outpath", required=True, help="Path to save closest .npz")
    parser.add_argument("--use_mps", action="store_true", help="Use Apple MPS backend")
    args = parser.parse_args()
    main(args)
