import os
import numpy as np
from PIL import Image
import torch
import math
from torch.autograd import Variable
from torchvision import transforms

import utils
import filters
import LRnet

torch.cuda.set_device(1)
norm_flag = False
make_lr_flag = True

refdir = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/images/banckmark"
lrdir = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/images/style-images/SR"

modelname = 'condition_SRnet_L_len5_ang33_g2.5_unorm_US_l1.model'
modeldir = os.path.join("/home/wcd/Projects/Pytorch-examples/fast_neural_style/neural_style/ckpt",modelname)

netdict = torch.load(modeldir)
opt = netdict["args"]

net = netdict["net"].eval()
net.load_state_dict(netdict["state_dict"])
net.cuda()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda X: X.mul(255))
])


def img2Variable(IMG):
    IMG = transform(IMG)
    IMG = IMG.unsqueeze(0)
    IMG = Variable(IMG, volatile=True)
    return IMG


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def psnr(im1, im2):
    MSE = mse(im1, im2)
    if MSE < 0.00000001:
        psnr = 10000
    else:
        psnr = 20 * np.log10(255) - 10 * np.log10(MSE)
    return psnr


def save_image(filename, data, mode):
    if mode is 'YCbCr':
        img = data.clone().numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        Y = Image.fromarray(img[:, :, 0], 'L')
        Cb = Image.fromarray(img[:, :, 1], 'L')
        Cr = Image.fromarray(img[:, :, 2], 'L')
        img = Image.merge('YCbCr', [Y, Cb, Cr]).convert('RGB')
    elif mode is 'RGB':
        img = data.clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        img = Image.fromarray(img, 'RGB')
    elif mode is 'L':
        img = data.squeeze(0).clone().clamp(0, 255).numpy().astype("uint8")
        img = Image.fromarray(img, 'L')
    img.save(filename)

def main1():

    mode = opt.mode
    params_min=[0.5,3,0]
    params_max=[2.5,8,145]
    params_num=[5,5,8]
    gausslist = np.linspace(params_min[0],params_max[0],params_num[0])
    motion_lenlist = np.linspace(params_min[1],params_max[1],params_num[1])
    motion_anglelist = np.linspace(params_min[2],params_max[2],params_num[2])
    avg_psnr_array = np.ndarray(params_num)
    avg_addpsnr_array = np.ndarray(params_num)
    num = 1
    print("|no.\t|gauss\t|motion_len\t|motion_ang\t|avg PSNR increament\t|avg PSNR\t|")
    for i,g in enumerate(gausslist):
        for j,len in enumerate(motion_lenlist):
            for k,ang in enumerate(motion_anglelist):
                avg_psnr, avg_addpsnr = culpsnr(refdir,mode, len, ang, g,num,printflag=False)
                avg_psnr_array[i,j,k] = avg_psnr
                avg_addpsnr_array[i,j,k] = avg_addpsnr
                num+=1
    import scipy.io as sio
    sio.savemat('data.mat',
                {'avg_psnr_array': avg_psnr_array, 'avg_addpsnr_array': avg_addpsnr_array, 'gausslist': gausslist,
                 'motion_lenlist': motion_lenlist, 'motion_anglelist':motion_anglelist})

    return

def main():

    mode = opt.mode
    params_min=[0.5,3,5]
    params_max=[2.5,8,85]
    params_num=[10,5,8]
    len_max = 7
    gauss_max = 2.5
    gausslist = np.linspace(params_min[0],params_max[0],params_num[0])
    motion_lenlist = np.linspace(params_min[1],params_max[1],params_num[1])
    motion_anglelist = np.linspace(params_min[2],params_max[2],params_num[2])
    num = 1
    print("|no.\t|gauss\t|motion_len\t|motion_ang\t|avg PSNR increament\t|avg PSNR\t|")

    for j,len in enumerate(motion_lenlist):
        for k,ang in enumerate(motion_anglelist):
            culpsnr(refdir,mode, len, ang, 0,num,printflag=False,len_max=len_max,gauss_max=gauss_max)
            num+=1

    for i,g in enumerate(gausslist):
        culpsnr(refdir, mode, 0.0, 0.0, g, num, printflag=False,len_max=len_max,gauss_max=gauss_max)
        num += 1

    return

def culpsnr(refdir,mode,motion_len, motion_angel,gauss,num,printflag=False,gauss_max=1,len_max=1):
    argstr="len_{:d}ang_{:d}g_{:.1f}".format(int(motion_len),int(motion_angel),gauss)

    outdir = os.path.join(refdir,argstr)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        os.mkdir(os.path.join(os.path.join(outdir,"LR")))
        os.mkdir(os.path.join(os.path.join(outdir,"SR")))

    if printflag:
        print("save dir: " + outdir)
        print("|img-name \t|blurPSNR | PSNR |")

    addpsnr_sum = 0
    psnr_sum = 0
    cnt = 0

    for img in os.listdir(refdir):
        hrIMGdir = os.path.join(refdir, img)
        if make_lr_flag is True:  # 自制lr，保存lr
            lrIMGdir = os.path.join(os.path.join(outdir, "LR"), img)
        else:  # 非自制lr 不用保存
            lrIMGdir = os.path.join(lrdir, img)
        srIMGdir = os.path.join(os.path.join(outdir, "SR"), img)

        if utils.is_image_file(img):
            hrIMG = utils.load_HR_image(hrIMGdir, size=None, mode=mode)
            motion_kernel, motion_anchor = filters.motion_kernel(motion_len, motion_angel)
            lrIMG = utils.HR2LR(hrIMG, motion_kernel, motion_anchor, gauss)
            motion_x = math.cos(math.radians(motion_angel))*motion_len
            motion_y = math.sin(math.radians(motion_angel))*motion_len
            if make_lr_flag is True:
                lrIMG.save(lrIMGdir)
            lr = img2Variable(lrIMG).cuda()
            c = Variable(LRnet.create_condition(1, torch.from_numpy(
                np.array([gauss, motion_x, motion_y])).unsqueeze(0), h=lr.size()[2],
                                                w=lr.size()[3],gauss_max=gauss_max,
                                           motion_x_max=len_max, motion_y_max=len_max), volatile=True).cuda()

            if norm_flag is True:
                lr = utils.normalize_batch(lr)
            sr = net(lr, c)
            if norm_flag is True:
                sr = utils.unnormalize_batch(sr)
            sr = sr.cpu().data.squeeze(0)
            save_image(srIMGdir, sr, mode)
            SRPSNR = psnr(sr.squeeze(0).clamp(0, 255).numpy(), np.array(hrIMG))
            LRPSNR = psnr(lr.data.cpu().squeeze(0).squeeze(0).clamp(0, 255).numpy(), np.array(hrIMG))

            addpsnr_sum += SRPSNR - LRPSNR
            psnr_sum += SRPSNR
            cnt +=1
            if printflag:
                print("|" + img + "\t |{:.2f}\t|{:.2f}|".format(LRPSNR, SRPSNR - LRPSNR))
    avg_psnr = psnr_sum/cnt
    avg_addpsnr = addpsnr_sum/cnt
    print('|{:3d}\t|{:.1f}\t|{:d}\t|{:d}\t|{:.2f}\t|{:.2f}\t|'.format(num,gauss,int(motion_len),int(motion_angel),avg_addpsnr,avg_psnr))
    return avg_psnr,avg_addpsnr

if __name__ == "__main__":
    main()