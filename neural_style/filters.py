import math
import PIL
import cv2
import numpy as np
import sys


def motion_kernel_matlab(angle=0, length=0):
    length = max(1, length)

    half = (length - 1) // 2
    # print(length)
    phi = math.radians(angle % 180)
    cosphi = math.cos(phi)
    sinphi = math.sin(phi)
    xsign = np.sign(cosphi)
    linewdt = 1
    eps = sys.float_info.epsilon
    # for 0,90

    sx = np.fix(half * cosphi + linewdt * xsign - length * eps)
    sy = np.fix(half * sinphi + linewdt - length * eps)
    # print([sx,sy])
    [x, y] = np.meshgrid(np.arange(0, sx + xsign, xsign), np.arange(0, sy + 1, 1))
    # print(x)
    # print(y)
    # define shortest distance from a pixel to the rotated line
    dist2line = (y * cosphi - x * sinphi)  # distance perpendicular to the line
    rad = np.sqrt(x ** 2 + y ** 2)
    # find points beyond the line's end-point but within the line width
    lastpix = (rad >= half) & (np.abs(dist2line) <= linewdt)
    # distance to the line's end-point parallel to the line
    x2lastpix = half - np.abs((x[lastpix] + dist2line[lastpix] * sinphi) / cosphi)
    dist2line[lastpix] = np.sqrt(dist2line[lastpix] ** 2 + x2lastpix ** 2)
    dist2line = linewdt + eps - np.abs(dist2line)
    dist2line[dist2line < 0] = 0
    # print(dist2line)
    dh = dist2line.shape[0]
    dw = dist2line.shape[1]
    h = np.zeros((dh * 2 - 1, dw * 2 - 1))
    # % unfold half-matrix to the full size
    h[:dh, :dw] = np.rot90(dist2line, k=2)
    h[dh - 1:, dw - 1:] = dist2line
    h = h / (h.sum() + eps * length * length)

    if cosphi > 0:
        h = np.flipud(h)

    return h,(-1,-1)

# 生成卷积核和锚点
def motion_kernel(length=0, angle=0):
    #print([length,angle])
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

    elif angle < -90:  # 同理：往左下角移动
        psf1 = np.flipud(psf1)
        anchor = (0, psf1.shape[0] - 1)

    kernel = psf1 / psf1.sum()
    return kernel, anchor


def motion_kernel2(length=0, angle=0):
    L=length
    theta=angle
    kernel = np.zeros([L, L])
    x = np.arange(0, L, 1) - int(L / 2)
    X, Y = np.meshgrid(x, x)

    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if np.sqrt(X[i, j] ** 2 + Y[i, j] ** 2) < L / 2:
                kernel[i, j] = LineIntegral(theta, X[i, j] - 0.5, X[i, j] + 0.5, -Y[i, j] - 0.5, -Y[i, j] + 0.5)

    return kernel/kernel.sum(),(-1,-1)


def LineIntegral(theta, a, b, alpha, beta):
    # Theta : between 0 and 360
    TanTheta = np.tan(np.deg2rad(theta))
    L = 0
    # Checks
    if a > b:
        x = a
        a = b
        b = x
    if alpha > beta:
        x = alpha
        alpha = beta
        beta = x

    if (theta != 90 and theta != 270 and theta != 0 and theta != 180):  # non vertical case
        if (a >= 0 and alpha >= 0):
            if alpha <= TanTheta * a <= beta:  # pointing upward, case 1
                if alpha <= TanTheta * b <= beta:
                    L = np.sqrt((b - a) ** 2 + (TanTheta * b - TanTheta * a) ** 2)
                else:
                    L = np.sqrt((beta / TanTheta - a) ** 2 + (beta - TanTheta * a) ** 2)
            elif a <= alpha / TanTheta <= b:  # pointing upward, case 2
                if alpha <= TanTheta * b <= beta:
                    L = np.sqrt((b - alpha / TanTheta) ** 2 + (TanTheta * b - alpha) ** 2)
                else:
                    L = np.sqrt((beta / TanTheta - alpha / TanTheta) ** 2 + (beta - alpha) ** 2)

        elif (a >= 0 and beta <= 0):
            if alpha <= TanTheta * a <= beta:  # pointing downward, case 1
                if alpha <= TanTheta * b <= beta:
                    L = np.sqrt((b - a) ** 2 + (TanTheta * b - TanTheta * a) ** 2)
                else:
                    L = np.sqrt((alpha / TanTheta - a) ** 2 + (alpha - TanTheta * a) ** 2)
            elif a <= alpha / TanTheta <= b:  # pointing downward, case 2
                if alpha <= TanTheta * b <= beta:
                    L = np.sqrt((b - beta / TanTheta) ** 2 + (TanTheta * b - beta) ** 2)
                else:
                    L = np.sqrt((alpha / TanTheta - beta / TanTheta) ** 2 + (alpha - beta) ** 2)

        elif (a <= 0 and alpha >= 0):
            if alpha <= TanTheta * b <= beta:  # pointing upward, case 1
                if alpha <= TanTheta * a <= beta:
                    L = np.sqrt((b - a) ** 2 + (TanTheta * b - TanTheta * a) ** 2)
                else:
                    L = np.sqrt((beta / TanTheta - b) ** 2 + (beta - TanTheta * b) ** 2)
            elif a <= alpha / TanTheta <= b:  # pointing upward, case 2
                if alpha <= TanTheta * a <= beta:
                    L = np.sqrt((a - alpha / TanTheta) ** 2 + (TanTheta * a - alpha) ** 2)
                else:
                    L = np.sqrt((beta / TanTheta - alpha / TanTheta) ** 2 + (beta - alpha) ** 2)

        elif (b <= 0 and beta <= 0):
            if alpha <= TanTheta * b <= beta:  # pointing downward, case 1
                if alpha <= TanTheta * a <= beta:
                    L = np.sqrt((b - a) ** 2 + (TanTheta * b - TanTheta * a) ** 2)
                else:
                    L = np.sqrt((alpha / TanTheta - b) ** 2 + (alpha - TanTheta * b) ** 2)
            elif a <= beta / TanTheta <= b:  # pointing downward, case 2
                if alpha <= TanTheta * a <= beta:
                    L = np.sqrt((a - beta / TanTheta) ** 2 + (TanTheta * a - beta) ** 2)
                else:
                    L = np.sqrt((alpha / TanTheta - beta / TanTheta) ** 2 + (alpha - beta) ** 2)

        elif (a < 0 and b > 0 and alpha < 0 and beta > 0):
            if alpha <= TanTheta * a <= beta:
                if alpha <= TanTheta * b <= beta:
                    L = np.sqrt((b - a) ** 2 + (a * TanTheta - b * TanTheta) ** 2)
                else:
                    if (TanTheta * a < TanTheta * b):
                        L = np.sqrt((beta / TanTheta - a) ** 2 + (a * TanTheta - beta) ** 2)
                    else:
                        L = np.sqrt((alpha / TanTheta - a) ** 2 + (a * TanTheta - alpha) ** 2)
            else:
                if a <= alpha / TanTheta <= b:
                    if alpha <= TanTheta * b <= beta:
                        L = np.sqrt((b - alpha / TanTheta) ** 2 + (alpha - b * TanTheta) ** 2)
                    else:
                        L = np.sqrt((beta / TanTheta - alpha / TanTheta) ** 2 + (alpha - beta) ** 2)
                else:
                    L = np.sqrt((beta / TanTheta - b) ** 2 + (TanTheta * b - beta) ** 2)

    else:
        if (theta == 90 or theta == 270):
            if (a < 0 and b > 0):
                L = (beta - alpha) * (b - a)
        else:
            if (alpha < 0 and beta > 0):
                L = (beta - alpha) * (b - a)

    return L

'''

def motion_blur(img,kernel, anchor):
    if kernel.size>1:
        #print([kernel,anchor])
        img = cv2.filter2D(np.array(img), -1, kernel, anchor=anchor)
        return PIL.Image.fromarray(img)
    else:
        return img
'''

def motion_blur(img,kernel,anchor=(-1,-1)):
    if kernel.size>1:
        #print([kernel,anchor])
        img = cv2.filter2D(np.array(img), -1, kernel,anchor=anchor)
        return PIL.Image.fromarray(img)
    else:
        return img

def gauss_blur(img,sigma,kernel=5):
    img =  cv2.GaussianBlur(np.array(img),(kernel,kernel),sigma)
    return PIL.Image.fromarray(img)

def defocus_kernel_matlab(rad = 3):
    if rad<=0.5:
        kernel=np.ones((1,1),dtype=np.float32)
    else:
        crad  = int(math.ceil(rad-0.5))
        kernel = np.zeros((crad*2+1,crad*2+1),dtype = np.float32)
        crod = np.arange(-crad, crad+1, 1)
        [x,y]=np.meshgrid(crod,crod)
        biggerxid = np.abs(x)>np.abs(y)
        maxxy = np.abs(y)
        minxy = np.abs(x)
        maxxy[biggerxid]=np.abs(x)[biggerxid]
        minxy[biggerxid]=np.abs(y)[biggerxid]

        id1 = (rad**2<(maxxy+0.5)**2+(minxy-0.5)**2)
        m1 = id1*(minxy-0.5) + np.sqrt( (1-id1)*(rad**2-(maxxy+0.5)**2))
        id2 = (rad**2>(maxxy-0.5)**2+(minxy+0.5)**2)
        m2 = id2*(minxy+0.5) + np.sqrt( (1-id2)*(rad**2-(maxxy-0.5)**2))
        id3 = (rad**2 < (maxxy+0.5)**2 + (minxy+0.5)**2)
        id4 = (rad**2 > (maxxy-0.5)**2 + (minxy-0.5)**2)
        sgrid = (rad**2*(0.5*(np.arcsin(m2/rad) - np.arcsin(m1/rad)) + 0.25*(np.sin(2*np.arcsin(m2/rad)) - np.sin(2*np.arcsin(m1/rad)))) - (maxxy-0.5)*(m2-m1) + (m1-minxy+0.5))
        id = (id3 & id4) |((minxy==0)&(maxxy-0.5 < rad)&(maxxy+0.5>=rad))
        sgrid = sgrid * id
        sgrid = sgrid +((maxxy+0.5)**2 + (minxy+0.5)**2 < rad**2)
        sgrid[crad,crad] = min(math.pi*rad**2,math.pi/2)
        if ((crad>0) and (rad > crad-0.5) and (rad**2 < (crad-0.5)**2+0.25)):
            m1  = np.sqrt(rad**2 - (crad - 0.5)**2)
            m1n = m1/rad;
            sg0 = 2*(rad**2*(0.5*np.arcsin(m1n) + 0.25*np.sin(2*np.arcsin(m1n)))-m1*(crad-0.5))
            sgrid[2*crad+1,crad+1] = sg0
            sgrid[crad+1,2*crad+1] = sg0
            sgrid[crad+1,1]        = sg0
            sgrid[1,crad+1]        = sg0
            sgrid[2*crad,crad+1]   = sgrid[2*crad,crad+1] - sg0
            sgrid[crad+1,2*crad]   = sgrid[crad+1,2*crad] - sg0
            sgrid[crad+1,2]        = sgrid[crad+1,2]      - sg0
            sgrid[2,crad+1]        = sgrid[2,crad+1]      - sg0
        sgrid[crad,crad] = min(sgrid[crad,crad],1);
        kernel=sgrid/sgrid.sum()
    return kernel

def psf_blur(origimg,psfnp):

    # in:
    #   origimg: PIL Image
    #   psfnp: numpy kernel 0-255

    # out:
    #   psfnp: numpy norm kernel 0-?
    #   blurimg: PIL Image

    # conv2d:  orig np img conv with float normkernel
    blur = cv2.filter2D(np.array(origimg).astype(np.float32),-1,psfnp)
    # blur result turn back to PIL Image
    blur = PIL.Image.fromarray( blur.astype(np.uint8))

    return blur



    #原先用 from PIL import Image,ImageFilter

def fuseimg(weights,imgs):
    suming = np.zeros(np.array(imgs[0]).shape)
    for i in range(len(weights)):
        suming = suming + weights[i]*np.array(imgs[i])
    return PIL.Image.fromarray(suming.astype(np.uint8))