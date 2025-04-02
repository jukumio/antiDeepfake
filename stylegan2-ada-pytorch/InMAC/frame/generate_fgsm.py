import os
import torch
import numpy as np
import PIL.Image
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from training.networks import Generator
import dnnlib
import legacy

def force_fp32(module):
    for param in module.parameters():
        if param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
    for buf in module.buffers():
        if buf.dtype != torch.float32:
            buf.data = buf.data.to(torch.float32)

def fgsm_attack(G, w, epsilon, target_img_tensor, vgg16, device):
    w_adv = w.clone().detach().requires_grad_(True)
    synth_img = G.synthesis(w_adv, noise_mode='const')
    synth_img_resized = F.interpolate((synth_img + 1) * 127.5, size=(256, 256), mode='area')

    target_resized = F.interpolate(target_img_tensor.unsqueeze(0), size=(256, 256), mode='area')
    synth_feat = vgg16(synth_img_resized, resize_images=False, return_lpips=True)
    target_feat = vgg16(target_resized, resize_images=False, return_lpips=True)

    loss = (target_feat - synth_feat).square().sum()
    loss.backward()

    w_adv = w_adv + epsilon * w_adv.grad.sign()
    return w_adv.detach()

def load_target_tensor(image_path, resolution, device):
    img = PIL.Image.open(image_path).convert('RGB')
    img = img.resize((resolution, resolution), PIL.Image.LANCZOS)
    img_np = np.array(img, dtype=np.uint8)
    img_tensor = torch.tensor(img_np.transpose([2, 0, 1]), device=device).float()
    return img_tensor / 255. * 2 - 1  # normalize to [-1, 1]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--w', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--use-mps', action='store_true')
    args = parser.parse_args()

    device = torch.device('mps') if args.use_mps and torch.backends.mps.is_available() else torch.device('cpu')

    print(f'Loading network from {args.network}')
    with dnnlib.util.open_url(args.network) as f:
        data = legacy.load_network_pkl(f)
        G_raw = data['G_ema']

    init_kwargs = G_raw.init_kwargs
    init_kwargs['synthesis_kwargs']['num_fp16_res'] = 0
    init_kwargs['synthesis_kwargs']['conv_clamp'] = None
    G = Generator(**init_kwargs).eval().requires_grad_(False).to(device)
    G.load_state_dict(G_raw.state_dict())
    G = G.float()
    force_fp32(G)

    # Load target image
    target_tensor = load_target_tensor(args.target, G.img_resolution, device)

    # Load vgg
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Load W latent
    w_npz = np.load(args.w)
    w = torch.tensor(w_npz['w'], device=device).float()

    # FGSM Attack
    w_adv = fgsm_attack(G, w, args.epsilon, target_tensor, vgg16, device)

    # Generate image
    synth_img = G.synthesis(w_adv, noise_mode='const')
    synth_img = (synth_img + 1) * (255/2)
    synth_img = synth_img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

    os.makedirs(args.outdir, exist_ok=True)
    PIL.Image.fromarray(synth_img, 'RGB').save(os.path.join(args.outdir, 'fgsm_proj.png'))
    np.savez(os.path.join(args.outdir, 'fgsm_w.npz'), w=w_adv.cpu().numpy())
    print(f'FGSM image and latent saved to {args.outdir}')

if __name__ == '__main__':
    main()