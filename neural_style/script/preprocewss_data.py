import argparse
import os
import sys
import time
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
import visdom

images = []
imgdir = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/images/yang"
imgdir_pre = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/images/yang_pre"
def img2Variable(fname,mode):
    if mode is 'YCbCr':
        HR = utils.load_HR_image(fname,mode = 'YCbCr')
    elif mode is 'RGB':
        HR = utils.load_HR_image(fname, mode='RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda X: X.mul(255))
    ])
    HR = transform(HR)
    HR = HR.unsqueeze(0)
    HR = Variable(HR, volatile=True)
    return HR

def save_image(filename, data, mode):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    if mode is 'YCbCr':
        a = Image.fromarray(img[:, :, 0], 'L')
        b = Image.fromarray(img[:, :, 1], 'L')
        c = Image.fromarray(img[:, :, 2], 'L')
        img = Image.merge('YCbCr', [a, b, c]).convert('RGB')
    elif mode is 'RGB':
        img = Image.fromarray(img, 'RGB')
    img.save(filename)

if not os.path.exists(imgdir_pre):
    os.mkdir(imgdir_pre)
for img in os.listdir(imgdir):
    hrname = os.path.join(imgdir, img)
    lrsavename = os.path.join(imgdir_pre, img)
    if utils.is_image_file(img):
        images.append(img)
        hr = Image.open(hrname)
        h, w = hr.size
        hr = hr.crop((0, 0, h - h % 4, w - w % 4))
        hr.save(lrsavename)
        print("save: "+lrsavename)

