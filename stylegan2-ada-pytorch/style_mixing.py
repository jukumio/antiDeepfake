# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate style mixing image matrix using pretrained network pickle."""

import os
import re
from typing import List

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--rows', 'row_seeds', type=str, help='Random seeds or projected_w.npz file for image rows', required=True)
@click.option('--cols', 'col_seeds', type=num_range, help='Random seeds to use for image columns', required=True)
@click.option('--styles', 'col_styles', type=num_range, help='Style layer range', default='0-6', show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', type=str, required=True)
def generate_style_mix(
    network_pkl: str,
    row_seeds: str,
    col_seeds: List[int],
    col_styles: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)

    # 수정된 부분: col_seeds에 대한 W 벡터 생성
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in col_seeds])
    col_w = G.mapping(torch.from_numpy(all_z).to(device), None)
    w_avg = G.mapping.w_avg
    col_w = w_avg + (col_w - w_avg) * truncation_psi
    w_dict = {seed: w for seed, w in zip(col_seeds, list(col_w))}

    # projected_w 로드
    if row_seeds.endswith('.npz'):
        print(f'Loading projected W from "{row_seeds}"')
        projected_w = np.load(row_seeds)['w']
        proj_w = torch.from_numpy(projected_w).to(device)
    else:
        raise ValueError('Please provide a projected_w.npz file')

    # 원본 이미지 생성
    print('Generating original image...')
    img = G.synthesis(proj_w, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    orig_image = img[0].cpu().numpy()

    print('Generating style-mixed images...')
    mixed_images = []
    for col_seed in col_seeds:
        w = proj_w.clone()
        w[0, col_styles] = w_dict[col_seed][col_styles]
        img = G.synthesis(w, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        mixed_images.append(img[0].cpu().numpy())

    print('Saving images...')
    PIL.Image.fromarray(orig_image, 'RGB').save(f'{outdir}/original.png')
    for idx, image in enumerate(mixed_images):
        PIL.Image.fromarray(image, 'RGB').save(f'{outdir}/mixed_{col_seeds[idx]}.png')

    print('Saving image grid...')
    W = G.img_resolution
    H = G.img_resolution
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * 2), 'black')
    
    # 첫 번째 행: 원본 이미지
    canvas.paste(PIL.Image.fromarray(orig_image, 'RGB'), (0, 0))
    
    # 두 번째 행: 스타일 혼합 이미지들
    for idx, image in enumerate(mixed_images):
        canvas.paste(PIL.Image.fromarray(image, 'RGB'), (W * (idx + 1), 0))

    canvas.save(f'{outdir}/grid.png')

if __name__ == "__main__":
    generate_style_mix() # pylint: disable=no-value-for-parameter