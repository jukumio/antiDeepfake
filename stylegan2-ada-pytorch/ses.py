import torch
from dnnlib.util import open_url

# VGG16 모델 로드
url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
device = torch.device('cuda')
with open_url(url) as f:
    vgg16 = torch.jit.load(f).eval().to(device)

# 임의의 입력 이미지
dummy_image = torch.randn(1, 3, 256, 256).to(device)  # [Batch, Channels, Height, Width]

# VGG16 출력
output = vgg16(dummy_image, resize_images=False, return_lpips=True)
print(f"VGG16 Output Shape: {output.shape}")
