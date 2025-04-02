import os
import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image
from time import perf_counter
import click

from training.networks import Generator
import dnnlib
import legacy

# -------------------------- Projection Core --------------------------
def project(G, target, *, num_steps=500, device, w_init=None, verbose=False):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)
    G = G.to(device).eval().requires_grad_(False)

    def logprint(*args):
        if verbose:
            print(*args)

    # Compute W stats
    logprint(f"Computing W midpoint and stddev using 10000 samples...")
    z_samples = np.random.randn(10000, G.z_dim).astype(np.float32)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_avg = w_samples.mean(0, keepdim=True)
    w_std = ((w_samples - w_avg) ** 2).sum() ** 0.5

    # Init latent
    if w_init is not None:
        assert w_init.ndim == 3 and w_init.shape[0] == 1, "w_init must be shape [1, num_ws, w_dim]"
        w_opt = w_init.clone().detach().requires_grad_(True)
    else:
        w_avg_plus = w_avg.repeat([1, G.synthesis.num_ws, 1])
        w_opt = w_avg_plus.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([w_opt], lr=0.1)

    # LPIPS feature extractor
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    target = target.unsqueeze(0).to(device)
    if target.shape[2] > 256:
        target = F.interpolate(target, size=(256, 256), mode='area')
    target_features = vgg16(target, resize_images=False, return_lpips=True)

    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    for step in range(num_steps):
        synth = G.synthesis(w_opt, noise_mode='const')
        if synth.shape[2] > 256:
            synth = F.interpolate(synth, size=(256, 256), mode='area')
        synth_features = vgg16(synth, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        optimizer.zero_grad(set_to_none=True)
        dist.backward()
        optimizer.step()

        w_out[step] = w_opt.detach()[0]
        logprint(f"Step {step + 1}/{num_steps} | Loss: {float(dist):.4f}")

    return w_out

# -------------------------- Main CLI --------------------------
@click.command()
@click.option('--network', required=True, help='Network pickle file')
@click.option('--target', required=True, help='Target image file')
@click.option('--w-init', required=True, help='Initial projected_w.npz path')
@click.option('--num-steps', default=500, help='Refinement steps')
@click.option('--outdir', required=True, help='Output directory')
@click.option('--use-mps', is_flag=True, help='Use MPS backend')
@click.option('--save-video', type=bool, default=False, help='Save video')

def run_refine(network, target, w_init, num_steps, outdir, use_mps, save_video):
    device = torch.device('mps') if use_mps and torch.backends.mps.is_available() else torch.device('cpu')
    print(f"[INFO] Using device: {device}")

    with dnnlib.util.open_url(network) as f:
        data = legacy.load_network_pkl(f)
    G_raw = data['G_ema']
    init_kwargs = G_raw.init_kwargs
    init_kwargs['synthesis_kwargs']['num_fp16_res'] = 0
    init_kwargs['synthesis_kwargs']['conv_clamp'] = None

    G = Generator(**init_kwargs)
    G.load_state_dict(G_raw.state_dict())
    G = G.eval().requires_grad_(False).to(device).float()

    # Load target
    target_pil = PIL.Image.open(target).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s)//2, (h - s)//2, (w + s)//2, (h + s)//2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_np = np.array(target_pil, dtype=np.uint8)
    target_tensor = torch.tensor(target_np.transpose(2, 0, 1), dtype=torch.float32, device=device)

    # Load w_init
    print(f"[INFO] Loading initial w from {w_init}")
    w_np = np.load(w_init)['w']
    w_tensor = torch.tensor(w_np, dtype=torch.float32, device=device)

    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=target_tensor,
        device=device,
        num_steps=num_steps,
        w_init=w_tensor,
        verbose=True
    )
    print(f"[INFO] Refinement completed in {perf_counter() - start_time:.1f}s")

    os.makedirs(outdir, exist_ok=True)
    final_w = projected_w_steps[-1]
    np.savez(os.path.join(outdir, 'refined_w.npz'), w=final_w.unsqueeze(0).cpu().numpy())

    synth = G.synthesis(final_w.unsqueeze(0), noise_mode='const')
    synth = (synth + 1) * 127.5
    synth = synth.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth, 'RGB').save(os.path.join(outdir, 'refined.png'))

if __name__ == '__main__':
    run_refine()