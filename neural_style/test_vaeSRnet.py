
from PIL import Image

import math
from torch.autograd import Variable
from torchvision import transforms

import os
import torch
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
from train_vaeSRnet import blindVAE
from options import vaeSROption
import utils
import filters

torch.cuda.set_device(1)
norm_flag = False
make_lr_flag = True

refdir = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/images/banckmark/vae"
lrdir = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/images/style-images/SR"

modelname = 'come4_st1760000_vaeSR_gauss_3.0_mang_89_mlen8_SRIN_block20_PBN_block8.model'
modeldir = os.path.join("/home/wcd/Projects/Pytorch-examples/fast_neural_style/neural_style/ckpt",modelname)

netdict = torch.load(modeldir)
opt = vaeSROption()
opt.seperate_model = modeldir
opt.batch_size = 1


net = blindVAE(opt,isTrain=False,isSeperate=False,fromPretrain=False)
net.load_network(modeldir)

transform = transforms.Compose([
    transforms.ToTensor(),
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


def save_image(filename, data, mode): #from cxhxw tensor
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
    return img

def main():

    mode = opt.color_mode
    params_min=[0,3,10]
    params_max=[3.0,8,80]
    params_num=[10,6,8]
    len_max = opt.motion_len_max
    gauss_max = opt.gauss_max
    gausslist = np.linspace(params_min[0],params_max[0],params_num[0])
    motion_lenlist = np.linspace(params_min[1],params_max[1],params_num[1])
    motion_anglelist = np.linspace(params_min[2],params_max[2],params_num[2])
    num = 1
    print("|no.\t|gauss\t|motion_len\t|motion_ang\t|avg PSNR increament\t|avg PSNR\t|")
    allpsnr_sun=0
    for j,len in enumerate(motion_lenlist):
        for k,ang in enumerate(motion_anglelist):
            sun,_,_ = culout(refdir,mode, len, ang, 0,num,printflag=False,len_max=len_max,gauss_max=gauss_max)
            allpsnr_sun+=sun
            num+=1
    for i, g in enumerate(gausslist):
        sun, _, _ =culout(refdir, mode, 0, 0, g, num, printflag=False, len_max=len_max, gauss_max=gauss_max)
        allpsnr_sun += sun
        num += 1
    print('avg:{:.4f}'.format(allpsnr_sun/num))
    return

def culout(refdir,mode,motion_len, motion_angel,gauss,num,printflag=False,gauss_max=1,len_max=1):
    argstr="len_{:d}ang_{:d}g_{:.1f}".format(int(motion_len),int(motion_angel),gauss)
    motion_x = math.cos(math.radians(motion_angel)) * motion_len
    motion_y = math.sin(math.radians(motion_angel)) * motion_len
    ptensor=torch.from_numpy(np.array([gauss,motion_x,motion_y]).astype(np.float32)).unsqueeze(0)

    outdir = os.path.join(refdir, argstr)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        os.mkdir(os.path.join(os.path.join(outdir, "LR")))
        os.mkdir(os.path.join(os.path.join(outdir, "SR")))

    if printflag:
        print("save dir: " + outdir)
        print("|img-name \t|blurPSNR | PSNR |")

    addpsnr_sum = 0
    psnr_sum = 0
    cnt = 0
    allpsnr_sun =0
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
            if make_lr_flag is True:
                lrIMG.save(lrIMGdir)
            lrtensor = transform(lrIMG).unsqueeze(0)
            lrtensor = (lrtensor-0.5)/0.5
            hrtensor = transform(hrIMG).unsqueeze(0)
            net.set_input([hrtensor,lrtensor,ptensor])
            net.validate()
            sr = net.SR
            if norm_flag is True:
                sr = utils.unnormalize_batch(sr)
            sr = sr.cpu().data.squeeze(0).clamp(-1,1)
            sr = (sr+1)/2.0 *255.0

            srIMG = save_image(srIMGdir, sr, mode)
            SRPSNR = psnr(np.array(srIMG), np.array(hrIMG))
            LRPSNR = psnr(np.array(lrIMG), np.array(hrIMG))

            addpsnr_sum += SRPSNR - LRPSNR
            psnr_sum += SRPSNR
            allpsnr_sun+=SRPSNR
            cnt +=1
            if printflag:
                print("|" + img + "\t |{:.2f}\t|{:.2f}|".format(LRPSNR, SRPSNR - LRPSNR))
    avg_psnr = psnr_sum / cnt
    avg_addpsnr = addpsnr_sum / cnt
    print('|{:3d}\t|{:.1f}\t|{:d}\t|{:d}\t|{:.2f}\t|{:.2f}\t|'.format(num, gauss, int(motion_len), int(motion_angel),
                                                                      avg_addpsnr, avg_psnr))
    return allpsnr_sun,avg_psnr, avg_addpsnr

if __name__ == "__main__":
    main()