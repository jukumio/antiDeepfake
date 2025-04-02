import torch

weights = torch.load("xception/model/xception-b5690688.pth", map_location="cpu")
print(weights.keys())
