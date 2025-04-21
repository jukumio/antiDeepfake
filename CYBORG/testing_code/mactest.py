import sys
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import csv
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('/Users/juheon/Desktop/DE_FAKE/capstone/CYBORG')
from xception.network.models import model_selection

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    device = torch.device('cpu')  # CUDA 대신 CPU 사용
    parser.add_argument('-imageFolder', default='Data', type=str)
    parser.add_argument('-modelPath', default='xception/model/xception-b5690688.pth', type=str)
    parser.add_argument('-output_scores', default='outputs/output_scores.csv', type=str)
    parser.add_argument('-network', default="xception", type=str)
    args = parser.parse_args()

    # output_scores 디렉토리 만들기
    output_dir = os.path.dirname(args.output_scores)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load model weights
    weights = torch.load(args.modelPath, map_location=device)

    if args.network == "resnet":
        im_size = 224
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif args.network == "inception":
        im_size = 299
        model = models.inception_v3(pretrained=True, aux_logits=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif args.network == "xception":
        im_size = 299
        model, *_ = model_selection(modelname='xception', num_out_classes=2)
    else:  # DenseNet
        im_size = 224
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)

    model.load_state_dict(weights, strict=False)
    model = model.to(device)
    model.eval()

    # Image transform
    if args.network == "xception":
        transform = transforms.Compose([
            transforms.Resize([im_size, im_size]),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize([im_size, im_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    imagesScores = []
    sigmoid = nn.Sigmoid()

    # Load all image files from the folder
    imageFiles = glob.glob(os.path.join(args.imageFolder, '*.png'))

    for imgFile in tqdm(imageFiles):
        image = Image.open(imgFile).convert('RGB')
        transformImage = transform(image)
        image.close()

        transformImage = transformImage[0:3, :, :].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(transformImage)

        PAScore = sigmoid(output).detach().cpu().numpy()[0, 1]  # float 값 하나만 추출
        filename = os.path.basename(imgFile)
        imagesScores.append([filename, float(PAScore)])  # 확실하게 float로 저장


    # Save output scores to CSV
    with open(args.output_scores, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['filename', 'score'])  # 헤더
        writer.writerows(imagesScores)
