import copy
import os
from time import perf_counter

import click
import cv2
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy
import face_alignment

def create_face_mask(image_tensor: torch.Tensor, device: torch.device, size=256) -> torch.Tensor:
    fa = face_alignment.FaceAlignment('2D', device=str(device))
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    landmarks_list = fa.get_landmarks(image_np)
    if landmarks_list is None:
        return torch.ones((1, size, size), device=device)

    mask = np.ones((size, size), dtype=np.float32)
    for landmarks in landmarks_list:
        def draw_circle(center, radius, weight):
            y, x = np.ogrid[:size, :size]
            cy, cx = int(center[1]), int(center[0])
            dist = (x - cx) ** 2 + (y - cy) ** 2
            circle_mask = dist <= radius ** 2
            mask[circle_mask] += weight

        left_eye_center = landmarks[36:42].mean(axis=0)
        right_eye_center = landmarks[42:48].mean(axis=0)
        nose_center = landmarks[27:36].mean(axis=0)
        mouth_center = landmarks[48:68].mean(axis=0)

        draw_circle(left_eye_center, radius=10, weight=1.0)
        draw_circle(right_eye_center, radius=10, weight=1.0)
        draw_circle(nose_center, radius=10, weight=0.5)
        draw_circle(mouth_center, radius=12, weight=1.0)

    mask = np.clip(mask, 1.0, 3.0)
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
    return mask_tensor

def project(G, target: torch.Tensor, *, num_steps=800, w_avg_samples=10000, initial_learning_rate=0.1,
            initial_noise_factor=0.05, lr_rampdown_length=0.25, lr_rampup_length=0.05,
            noise_ramp_length=0.75, regularize_noise_weight=1e5, verbose=False, device: torch.device):

    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    logprint(f'Computing W+ midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim).astype(np.float32)
    z_samples_tensor = torch.from_numpy(z_samples).to(device=device, dtype=torch.float32)
    print(f"[DEBUG] G.mapping.num_ws = {G.mapping.num_ws}")


    w_samples = G.mapping(z_samples_tensor, None)
    w_samples = w_samples.cpu().numpy().astype(np.float32)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)
    w_std = (np.sum((w_samples - w_avg[:, :1, :]) ** 2) / w_avg_samples) ** 0.5

    latent_path = os.getenv("HYPERSTYLE_LATENT")
    if latent_path and os.path.exists(latent_path):
        print(f"Loading initial latent from: {latent_path}")
        w_tensor = torch.load(latent_path, map_location='cpu')
        if isinstance(w_tensor, dict):
            if 'latent_avg' in w_tensor:
                w_tensor = w_tensor['latent_avg']
            else:
                raise ValueError(f"Expected key 'latent_avg' not found. Available keys: {list(w_tensor.keys())}")
        if not torch.is_tensor(w_tensor):
            raise ValueError("Loaded latent is not a tensor")

        if w_tensor.ndim == 2:
            w_tensor = w_tensor.unsqueeze(0)
        elif w_tensor.ndim == 1:
            w_tensor = w_tensor.unsqueeze(0).unsqueeze(1).repeat(1, G.mapping.num_ws, 1)

        w_avg_plus = w_tensor.detach().cpu().numpy().astype(np.float32)
    else:
        w_avg_plus = np.tile(w_avg[:, :1, :], (1, G.mapping.num_ws, 1)).astype(np.float32)

    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    target_images = target.unsqueeze(0).to(device=device, dtype=torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    face_mask = create_face_mask(target_images[0], device=device, size=target_images.shape[2])

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
        ws = w_opt + w_noise
        
        print(f"[DEBUG] G.synthesis.num_ws = {G.synthesis.num_ws}")
        print(f"[DEBUG] ws.shape = {ws.shape}")
        ws = ws.to(dtype=torch.float32, device=device)  

        print(f"Step {step}: ws shape = {ws.shape}, dtype = {ws.dtype}, device = {ws.device}")

        # Debugging shape
        print(f"Step {step}: ws shape = {ws.shape}, expected = [batch, {G.mapping.num_ws}, {G.w_dim}]")
        assert ws.ndim == 3 and ws.shape[1] == G.mapping.num_ws, f"ws shape invalid: {ws.shape}"

        with torch.no_grad():
            synth_images = G.synthesis(ws, noise_mode='const')

        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()
        pixel_loss = F.mse_loss(synth_images * face_mask, target_images * face_mask)

        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        loss = dist + reg_loss * regularize_noise_weight + 0.1 * pixel_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} pixel {pixel_loss:<4.2f} loss {float(loss):<5.2f}')

        w_out[step] = w_opt.detach()[0]

        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps', help='Number of optimization steps', type=int, default=800, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=300, show_default=True)
@click.option('--save-video', help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
def run_projection(network_pkl, target_fname, outdir, save_video, seed, num_steps):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cpu')

    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)

    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
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
    projected_w = projected_w_steps[-50:].mean(0)

    # Sanity check
    assert projected_w.ndim == 2, f"Expected projected_w with shape [num_ws, w_dim], got {projected_w.shape}"
    assert projected_w.shape[0] == G.synthesis.num_ws, f"Mismatch in num_ws: expected {G.synthesis.num_ws}, got {projected_w.shape[0]}"

    projected_w = projected_w.to(device)

    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')

    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

if __name__ == "__main__":
    run_projection()
