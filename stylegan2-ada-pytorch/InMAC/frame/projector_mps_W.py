# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

"""Project given image to the latent space of pretrained network pickle using W+ latent space on macOS/MPS with float32 conversion from original StyleGAN2-ADA models."""

import sys
import copy
import os
from time import perf_counter

import click
import cv2
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dnnlib
import legacy
from training.networks import Generator

def force_fp32(module: torch.nn.Module):
    for _, param in module.named_parameters(recurse=True):
        if param.data.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
    for _, buf in module.named_buffers(recurse=True):
        if buf.dtype != torch.float32:
            buf.data = buf.data.to(torch.float32)

from projector import project  # Use original projector logic with single-vector init

@click.command()
@click.option('--network', 'network_pkl', help='Original network pickle filename (.pkl)', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps', help='Number of optimization steps', type=int, default=1000)
@click.option('--seed', help='Random seed', type=int, default=300)
@click.option('--save-video', help='Save mp4 video', type=bool, default=True)
@click.option('--outdir', help='Output directory', required=True, metavar='DIR')
@click.option('--use-mps', is_flag=True, help='Use Apple MPS backend (default is CPU)')
def run_projection(network_pkl, target_fname, outdir, save_video, seed, num_steps, use_mps):
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f'[INFO] Using seed: {seed}')
    print(f'Loading network from "{network_pkl}"')

    device = torch.device('mps') if use_mps and torch.backends.mps.is_available() else torch.device('cpu')

    with dnnlib.util.open_url(network_pkl) as f:
        data = legacy.load_network_pkl(f)
        G_raw = data['G_ema']

    init_kwargs = copy.deepcopy(G_raw.init_kwargs)
    init_kwargs['synthesis_kwargs']['num_fp16_res'] = 0
    init_kwargs['synthesis_kwargs']['conv_clamp'] = None

    G = Generator(**init_kwargs)
    G.load_state_dict(G_raw.state_dict())
    G = G.eval().requires_grad_(False).to(device).float()
    force_fp32(G)

    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s)//2, (h - s)//2, (w + s)//2, (h + s)//2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device, dtype=torch.float32),
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    os.makedirs(outdir, exist_ok=True)
    if save_video:
        try:
            frame_size = (target_uint8.shape[1] * 2, target_uint8.shape[0])
            video_path = os.path.join(outdir, 'proj.mp4')
            video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, frame_size)
            print(f'Saving video: {video_path}')

            target_bgr = cv2.cvtColor(target_uint8, cv2.COLOR_RGB2BGR)
            for projected_w in projected_w_steps:
                synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
                synth_image = (synth_image + 1) * (255/2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                synth_bgr = cv2.cvtColor(synth_image, cv2.COLOR_RGB2BGR)
                video.write(np.concatenate([target_bgr, synth_bgr], axis=1))
            video.release()
        except Exception as e:
            print(f'[Warning] Failed to save video: {e}')

    target_pil.save(os.path.join(outdir, 'target.png'))
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(os.path.join(outdir, 'proj.png'))

    npz_path = os.path.join(outdir, 'projected_w.npz')
    np.savez(npz_path, w=projected_w.unsqueeze(0).cpu().numpy())
    if os.path.isfile(npz_path):
        print(f"[✔] projected_w.npz saved successfully at: {npz_path}")
    else:
        print(f"[✘] Failed to save projected_w.npz at: {npz_path}")

if __name__ == '__main__':
    run_projection()
