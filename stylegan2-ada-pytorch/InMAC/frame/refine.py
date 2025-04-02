import os
import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image
from time import perf_counter
import click
import cv2
import copy

from training.networks import Generator
import dnnlib
import legacy

# -------------------------- Projection Core --------------------------
def project(G, target, *, device, num_steps=500, w_init=None, initial_lr=0.05, betas=(0.9, 0.999), lpips_weight=1.0, reg_noise_weight=1e5, noise_mode='const', verbose=False):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    # W statistics
    logprint("Computing W midpoint and stddev using 10000 samples...")
    z_samples = np.random.randn(10000, G.z_dim).astype(np.float32)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)
    w_avg = w_samples.mean(0, keepdim=True)

    w_opt = w_init.clone().detach().requires_grad_(True)

    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}
    for buf in noise_bufs.values():
        buf.requires_grad = True

    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), lr=initial_lr, betas=betas)

    with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt') as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    target_images = target.unsqueeze(0).to(torch.float32)
    target_features = vgg16(F.interpolate(target_images, size=(256, 256), mode='area'), resize_images=False, return_lpips=True)

    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)

    for step in range(num_steps):
        synth_images = G.synthesis(w_opt, noise_mode=noise_mode)
        synth_images_resized = F.interpolate((synth_images + 1) * (255 / 2), size=(256, 256), mode='area')
        synth_features = vgg16(synth_images_resized, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum() * lpips_weight

        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]
            while True:
                reg_loss += (noise * torch.roll(noise, 1, dims=3)).mean().square()
                reg_loss += (noise * torch.roll(noise, 1, dims=2)).mean().square()
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, 2)

        loss = dist + reg_loss * reg_noise_weight

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        w_out[step] = w_opt.detach()[0]

        if step % 10 == 0 or step == num_steps - 1:
            logprint(f"Step {step+1}/{num_steps} | Loss: {loss.item():.4f}")

        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out

@click.command()
@click.option('--network', required=True, help='Network pickle file')
@click.option('--target', required=True, help='Target image file')
@click.option('--w-init', required=True, help='Initial projected_w.npz path')
@click.option('--num-steps', default=500, help='Refinement steps')
@click.option('--initial-lr', default=0.05, help='Initial learning rate')
@click.option('--betas', nargs=2, type=float, default=(0.95, 0.999), help='Adam optimizer betas')
@click.option('--lpips-weight', default=1.0, help='LPIPS loss weight')
@click.option('--reg-noise-weight', default=1e5, help='Noise regularization weight')
@click.option('--noise-mode', default='const', type=click.Choice(['const', 'random']), help='Noise mode for synthesis')
@click.option('--outdir', required=True, help='Output directory')
@click.option('--use-mps', is_flag=True, help='Use MPS backend')
@click.option('--save-video', is_flag=True, default=False, help='Save refinement video')
def run_refine(network, target, w_init, num_steps, initial_lr, betas, lpips_weight, reg_noise_weight, noise_mode, outdir, use_mps, save_video):
    device = torch.device('mps') if use_mps and torch.backends.mps.is_available() else torch.device('cpu')

    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].eval().to(device)

    target_pil = PIL.Image.open(target).convert('RGB').resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_tensor = torch.tensor(np.array(target_pil).transpose(2,0,1), dtype=torch.float32, device=device)

    w_tensor = torch.tensor(np.load(w_init)['w'], dtype=torch.float32, device=device)

    projected_w_steps = project(
        G, target_tensor, device=device, num_steps=num_steps, w_init=w_tensor,
        initial_lr=initial_lr, betas=betas, lpips_weight=lpips_weight,
        reg_noise_weight=reg_noise_weight, noise_mode=noise_mode, verbose=True
    )

    os.makedirs(outdir, exist_ok=True)
    final_w = projected_w_steps[-1]
    np.savez(os.path.join(outdir, 'refined_w.npz'), w=final_w.unsqueeze(0).cpu().numpy())

if __name__ == '__main__':
    run_refine()
