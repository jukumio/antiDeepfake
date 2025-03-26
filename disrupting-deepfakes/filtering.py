import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np

class PGDAttack:
    def __init__(self, model, device, epsilon=0.03, alpha=0.005, steps=10):
        """
        PGD 공격 초기화
        model: 공격할 얼굴 변환 모델 (MobileFaceSwap 등)
        device: 'cuda' 또는 'cpu'
        epsilon: 최대 공격 크기 (0.03 = 약 ±8/255)
        alpha: 스텝 크기 (0.005 = 약 ±1/255)
        steps: 공격 반복 횟수
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.device = device

    def attack(self, image_path, save_path):
        """PGD 공격을 수행하여 얼굴 변환 모델을 방해"""
        # 이미지 로드 및 전처리
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        image = Image.open(image_path).convert('RGB')
        x = transform(image).unsqueeze(0).to(self.device)  # 배치 차원 추가
        x_orig = x.clone().detach()

        # 모델에 대한 예측 수행 (MobileFaceSwap이 아닌 더미 모델 가능)
        x_adv = x.clone().detach().requires_grad_(True)
        for _ in range(self.steps):
            output = self.model(x_adv)  # 공격 대상 모델을 통과시킴
            loss = -F.mse_loss(output, self.model(x_orig))  # 원본과 다르게 변형하도록 손실 유도
            self.model.zero_grad()
            loss.backward()
            grad = x_adv.grad

            # 적대적 업데이트
            x_adv = x_adv + self.alpha * grad.sign()
            eta = torch.clamp(x_adv - x_orig, min=-self.epsilon, max=self.epsilon)
            x_adv = torch.clamp(x_orig + eta, min=-1, max=1).detach_().requires_grad_(True)

        # 변형된 이미지 저장
        x_adv = (x_adv * 0.5) + 0.5  # [-1,1] → [0,1] 변환
        save_image(x_adv, save_path)
        print(f"✅ PGD 공격된 이미지를 저장했습니다: {save_path}")



pgd = PGDAttack(model=torch.nn.Identity().to("cuda"), device="cuda")
# pgd.attack("input.jpg", "output_pgd.jpg")