# projector_with_weighted_mask_loss.py (StyleGAN projector with facial mask weighting)

import os
import copy
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import cv2
import click
from time import perf_counter

import dnnlib
import legacy
import face_alignment

from torchvision import transforms
from lpips import LPIPS  # Ensure lpips is installed: pip install lpips

def convert_module_to_float32(module):
    for name, param in module.named_parameters(recurse=True):
        if param is not None:
            param.data = param.data.float()
    for name, buffer in module.named_buffers(recurse=True):
        if buffer is not None:
            buffer.data = buffer.data.float()

def move_module_and_buffers_to_device(module, device):
    module.to(device)
    for name, param in module.named_parameters(recurse=True):
        if param is not None:
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad = param.grad.to(device)
    for name, buffer in module.named_buffers(recurse=True):
        if buffer is not None:
            try:
                setattr(module, name, buffer.to(device))
            except Exception:
                pass

def create_face_mask(image_tensor: torch.Tensor, device: torch.device, size=256) -> torch.Tensor:
    fa = face_alignment.FaceAlignment('2D', device='cpu')
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    landmarks_list = fa.get_landmarks(image_np)
    if landmarks_list is None:
        print("[WARNING] No faces were detected. Using default mask.")
        return torch.ones((1, size, size), device='cpu')

    mask = np.ones((size, size), dtype=np.float32)
    for landmarks in landmarks_list:
        def draw_circle(center, radius, weight):
            y, x = np.ogrid[:size, :size]
            cy, cx = int(center[1]), int(center[0])
            dist = (x - cx) ** 2 + (y - cy) ** 2
            circle_mask = dist <= radius ** 2
            mask[circle_mask] += weight

        draw_circle(landmarks[36:42].mean(axis=0), 10, 1.0)
        draw_circle(landmarks[42:48].mean(axis=0), 10, 1.0)
        draw_circle(landmarks[27:36].mean(axis=0), 10, 0.5)
        draw_circle(landmarks[48:68].mean(axis=0), 12, 1.0)

    mask = np.clip(mask, 1.0, 3.0)
    return torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

def project_with_weighted_lpips(
    G,
    target: torch.Tensor,
    *,
    num_steps=800,
    initial_learning_rate=0.1,
    device: torch.device,
    noise_mode='const',
):
    assert target.shape[0] == G.img_channels
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    convert_module_to_float32(G)
    move_module_and_buffers_to_device(G, device)

    percept = LPIPS(net='vgg').eval().to(device)

    z_samples = np.random.randn(10000, G.z_dim).astype(np.float32)
    z_samples = torch.from_numpy(z_samples).to(device)
    w_samples = G.mapping(z_samples, None)
    if w_samples.ndim == 2:
        w_samples = w_samples.unsqueeze(1).repeat(1, G.num_ws, 1)

    w_avg = w_samples.mean(dim=0, keepdim=True)
    w_std = w_samples.std(dim=0, keepdim=True)

    w_opt = w_avg.detach().clone().to(device).requires_grad_(True)  # [1, G.num_ws, G.w_dim]
    w_out = torch.zeros([num_steps, G.num_ws, G.w_dim], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt], lr=initial_learning_rate)

    noise_bufs = {name: buf for name, buf in G.synthesis.named_buffers() if 'noise_const' in name}
    for name, buf in noise_bufs.items():
        buf = buf.to(device)
        buf.requires_grad = True
        setattr(G.synthesis, name, buf)
        optimizer.add_param_group({"params": buf})

    target_image = target.unsqueeze(0).to(device, dtype=torch.float32)
    face_mask = create_face_mask(target, device='cpu', size=target.shape[1]).unsqueeze(0).to(device)

    for step in range(num_steps):
        ws_noise = torch.randn_like(w_opt) * w_std.to(device) * (1 - step / num_steps) ** 2
        ws = w_opt + ws_noise  # [1, G.num_ws, G.w_dim]

        synth_images = G.synthesis(ws, noise_mode=noise_mode)

        lpips_diff = percept(synth_images, target_image)
        lpips_weighted = (lpips_diff * face_mask).mean()

        reg_loss = sum(
            (v * torch.roll(v, 1, dims=2)).mean() ** 2 +
            (v * torch.roll(v, 1, dims=3)).mean() ** 2 for v in noise_bufs.values()
        )

        loss = lpips_weighted + reg_loss * 1e5
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[{device}] step {step:4d}: weighted_lpips {lpips_weighted:.4f}, loss {loss:.4f}")
        w_out[step] = ws.detach()[0]

    return w_out

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps', help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=300, show_default=True)
@click.option('--save-video', help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')

def run_projection(network_pkl, target_fname, outdir, save_video, seed, num_steps):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)

    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    start_time = perf_counter()
    projected_w_steps = project_with_weighted_lpips(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device, dtype=torch.float32),
        num_steps=num_steps,
        device=device,
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    os.makedirs(outdir, exist_ok=True)
    if save_video:
        try:
            frame_size = (target_uint8.shape[1] * 2, target_uint8.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = f'{outdir}/proj.mp4'
            video = cv2.VideoWriter(video_path, fourcc, 30, frame_size)
            print(f'Saving optimization progress video "{video_path}"')

            target_bgr = cv2.cvtColor(target_uint8, cv2.COLOR_RGB2BGR)

            for projected_w in projected_w_steps:
                synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
                synth_image = (synth_image + 1) * (255/2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                synth_image_bgr = cv2.cvtColor(synth_image, cv2.COLOR_RGB2BGR)
                combined_frame = np.concatenate([target_bgr, synth_image_bgr], axis=1)
                video.write(combined_frame)

            video.release()

        except Exception as e:
            print(f'Warning: video save failed, skipping... Error: {str(e)}')

    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

if __name__ == "__main__":
    run_projection()
