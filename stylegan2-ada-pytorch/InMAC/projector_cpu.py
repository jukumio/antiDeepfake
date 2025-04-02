# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

"""Project given image to the latent space of pretrained network pickle using W+ latent space on macOS/CPU with float32 conversion from original StyleGAN2-ADA models."""

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
    for name, param in module.named_parameters(recurse=True):
        if param.data.dtype != torch.float32:
            print(f"[FIX PARAM] {name} was {param.data.dtype}")
            param.data = param.data.to(torch.float32)
    for name, buf in module.named_buffers(recurse=True):
        if buf.dtype != torch.float32:
            print(f"[FIX BUFFER] {name} was {buf.dtype}")
            buf.data = buf.data.to(torch.float32)

def project(
    G,
    target: torch.Tensor,
    *,
    num_steps=1000,
    w_avg_samples=10000,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    regularize_noise_weight=1e5,
    verbose=False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = G.eval().requires_grad_(False).to(device)
    G = G.float()
    force_fp32(G)

    logprint(f'Computing W+ midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim).astype(np.float32)
    z_samples_tensor = torch.from_numpy(z_samples).to(device)
    w_samples = G.mapping(z_samples_tensor, None)
    w_samples = w_samples.cpu().numpy().astype(np.float32)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)
    w_std = (np.sum((w_samples - w_avg[:, :1, :]) ** 2) / w_avg_samples) ** 0.5

    w_avg_plus = np.tile(w_avg[:, :1, :], (1, G.synthesis.num_ws, 1)).astype(np.float32)
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    target_images = target.unsqueeze(0).to(device)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg_plus, dtype=torch.float32, device=device, requires_grad=True)
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).to(device=device, dtype=torch.float32).detach()
        synth_images = G.synthesis(ws, noise_mode='const')
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        w_out[step] = w_opt.detach()[0]

        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out

@click.command()
@click.option('--network', 'network_pkl', help='Original network pickle filename (.pkl)', required=True, default='/Users/juheon/Desktop/DE_FAKE/capstone/stylegan2-ada-pytorch/weights/ffhq.pkl')
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE', default='/Users/juheon/Desktop/DE_FAKE/capstone/mysource/smith.jpg')
@click.option('--num-steps', help='Number of optimization steps', type=int, default=1000)
@click.option('--seed', help='Random seed', type=int, default=300)
@click.option('--save-video', help='Save mp4 video', type=bool, default=True)
@click.option('--outdir', help='Output directory', required=True, metavar='DIR', default='/Users/juheon/Desktop/DE_FAKE/capstone/results')

def run_projection(network_pkl, target_fname, outdir, save_video, seed, num_steps):
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f'Loading network from "{network_pkl}"')

    device = torch.device('cpu')

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
    np.savez(os.path.join(outdir, 'projected_w.npz'), w=projected_w.unsqueeze(0).cpu().numpy())

if __name__ == '__main__':
    run_projection()