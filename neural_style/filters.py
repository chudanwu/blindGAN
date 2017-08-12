import math

import PIL
import cv2
import numpy as np


# 生成卷积核和锚点
def motion_kernel(length, angle):
    EPS = np.finfo(float).eps
    alpha = (angle - math.floor(angle / 180) * 180) / 180 * math.pi
    half = length / 2
    cosalpha = math.cos(alpha)
    sinalpha = math.sin(alpha)
    if cosalpha < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:
        xsign = 1
    psfwdt = 1;
    # 模糊核大小
    sx = int(math.fabs(length * cosalpha + psfwdt * xsign - length * EPS))
    sy = int(math.fabs(length * sinalpha + psfwdt - length * EPS))
    psf1 = np.zeros((sy, sx))

    # psf1是左上角的权值较大，越往右下角权值越小的核。
    # 这时运动像是从右下角到左上角移动
    for i in range(0, sy):
        for j in range(0, sx):
            psf1[i][j] = i * math.fabs(cosalpha) - j * sinalpha
            rad = math.sqrt(i * i + j * j)
            if rad >= half and math.fabs(psf1[i][j]) <= psfwdt:
                temp = half - math.fabs((j + psf1[i][j] * sinalpha) / cosalpha)
                psf1[i][j] = math.sqrt(psf1[i][j] * psf1[i][j] + temp * temp)
            psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j]);
            if psf1[i][j] < 0:
                psf1[i][j] = 0
    # 运动方向是往左上运动，锚点在（0，0）
    anchor = (0, 0)
    # 运动方向是往右上角移动，锚点一个在右上角
    # 同时，左右翻转核函数，使得越靠近锚点，权值越大
    if angle < 90 and angle > 0:
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, 0)
    elif angle > -90 and angle < 0:  # 同理：往右下角移动
        psf1 = np.flipud(psf1)
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, psf1.shape[0] - 1)

    elif anchor < -90:  # 同理：往左下角移动
        psf1 = np.flipud(psf1)
        anchor = (0, psf1.shape[0] - 1)

    kernel = psf1 / psf1.sum()
    return kernel, anchor

def motion_blur(img,kernel, anchor):
    img = cv2.filter2D(np.array(img), -1, kernel, anchor=anchor)
    return PIL.Image.fromarray(img)


def gauss_blur(img,sigma,kernel=5):
    img =  cv2.GaussianBlur(np.array(img),(kernel,kernel),sigma)
    return PIL.Image.fromarray(img)

def psf_blur(origimg,psfnp,size):

    # in:
    #   origimg: PIL Image
    #   psfnp: numpy kernel 0-255

    # out:
    #   psfnp: numpy norm kernel 0-?
    #   blurimg: PIL Image

    # normalize psf
    psfnp /= 255
    psfnp /= psfnp.sum()
    # conv2d:  orig np img conv with float normkernel
    blur = cv2.filter2D(np.array(origimg).astype(np.float32),-1,psfnp)
    # blur result turn back to PIL Image
    blur = PIL.Image.fromarray( blur.astype(np.uint8))

    h, w = size
    blur = blur.crop((0, 0, h - h % 4, w - w % 4))

    if size is not None:
        blur = blur.resize((size, size), origimg.ANTIALIAS)
    return psfnp,blur



    #原先用 from PIL import Image,ImageFilter