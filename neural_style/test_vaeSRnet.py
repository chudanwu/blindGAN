
from PIL import Image

import math
from torch.autograd import Variable
from torchvision import transforms

import os
import torch
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np

from options import vaeSROption
import utils
import filters


def tensor2img_L(tensor): #1xhxw -> hxw
    img = tensor.squeeze(0).clone().clamp(0, 255).numpy().astype("uint8")
    return Image.fromarray(img, 'L')

torch.cuda.set_device(0)
norm_flag = False
make_lr_flag = True
noparam = False

refdir = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/images/banckmark/vae"
lrdir = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/images/style-images/SR"

from train_block_vaeSRnet import blindVAE
modeldir = "/media/library/wcd/Models/Train_deblur/patch_in/condition_p/come10_st960000_block_vaeSR_gsig_3.0_glen_3_mang_90_mlen10_SRIN_block16_PBN_block8.model"
if noparam:
    modelname = 'sepe18_st1440000_nop_block_vaeSR_gsig_3.0_glen_3_mang_89_mlen8_SRIN_block4_PBN_block8.model'
    modeldir = os.path.join("/home/wcd/Desktop/to_neural_style/ckpt/non_params/block4/",modelname)
    modeldir = '/home/wcd/Desktop/to_neural_style/ckpt/non_params/block4/come18_st1440000_nop_block_vaeSR_gsig_3.0_glen_3_mang_89_mlen8_SRIN_block4_PBN_block8.model'

'''
from train_vaeSRnet import blindVAE
modelname = 'come18_st2960000_block_vaeSR_gsig_3.0_glen_3_mang_89_mlen8_SRIN_block4_PBN_block8.model'
modeldir = os.path.join("/home/wcd/Desktop/to_neural_style/ckpt/nonactivate_out/",modelname)
'''

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
    params_min=[0.0, 10, 2, 1]
    params_max=[3.0, 90, 10, 5]
    params_num=[ 10,  5, 5, 5]
    m_len_max = opt.motion_len_max
    g_sig_max = opt.gauss_sig_max
    d_rad_max = opt.defocus_rad_max
    g_siglist = np.linspace(params_min[0],params_max[0],params_num[0])
    m_anglelist = np.linspace(params_min[1],params_max[1],params_num[1])
    m_lenlist = np.linspace(params_min[2],params_max[2],params_num[2])
    d_radlist = np.linspace(params_min[3],params_max[3],params_num[3])
    num = 1
    print("|no.	|g_sig(len)	|m_ang	|m_len	|avg PSNR(+)	|avg PSNR	|")
    allpsnr_sun=0
    for j,len in enumerate(m_lenlist):
        for k,ang in enumerate(m_anglelist):
            sun,_,_ = culout(refdir,mode,num,g_sig_max,m_len_max,d_rad_max,
                             motion_angle=int(ang),motion_len=int(len),printflag=False)
            allpsnr_sun+=sun
            num+=1
    for k, sig in enumerate(g_siglist):
        sun, _, _ =culout(refdir, mode,num, g_sig_max,m_len_max,d_rad_max,
                          gauss_sig=sig,printflag=False)
        allpsnr_sun += sun
        num += 1
    for k, rad in enumerate(d_radlist):
        sun,_,_ = culout(refdir, mode,num, g_sig_max,m_len_max,d_rad_max,
                          defocus_rad=int(rad),printflag=False)
        allpsnr_sun += sun
        num += 1
    print('avg/num:{:.4f}'.format(allpsnr_sun/num))
    print('avg/pic:{:.4f}'.format(allpsnr_sun/(num*79)))
    return

def get_ptensor(h,w,g_sig_max,m_len_max,d_rad_max,g_sig=0,m_ang=0,m_len=0,d_rad=0):
    params = np.zeros((4, h, w), dtype=np.float32)
    params[0] = np.full((h,w), g_sig / g_sig_max)
    motion_x = m_len * math.cos(math.radians(m_ang))
    motion_y = m_len * math.sin(math.radians(m_ang))
    params[1] = np.full((h,w), motion_x / (m_len_max))
    params[2] = np.full((h,w), motion_y / (m_len_max))
    params[3] = np.full((h,w), d_rad / d_rad_max)
    return torch.from_numpy(params)

def culout(refdir,mode,num,g_sig_max,m_len_max,d_rad_max,gauss_sig=0, motion_angle=0,motion_len=0,defocus_rad=0,printflag=False):
    argstr="gsig_{:.1f}mlen_{:d}mang_{:d}drad_{:d}".format(gauss_sig,motion_len,motion_angle,defocus_rad)
    gauss_len = math.ceil(gauss_sig*6)
    gauss_len = int(gauss_len +1- gauss_len%2)
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
            hrnp = np.array(hrIMG)
            h, w = hrnp.shape
            lrnp = utils.HRnp2LRnp(hrnp,
                                   gauss_len=gauss_len,gauss_sig=gauss_sig,
                                   motion_angle=motion_angle,motion_len=motion_len,
                                   defocus_rad=defocus_rad,scale=None)
            lrIMG = Image.fromarray(lrnp)
            if make_lr_flag is True:
                lrIMG.save(lrIMGdir)
            lrtensor = transform(lrIMG).unsqueeze(0)
            lrtensor = (lrtensor-0.5)/0.5
            hrtensor = transform(hrIMG).unsqueeze(0)
            ptensor = get_ptensor(h,w, g_sig_max,m_len_max,d_rad_max,
                                  g_sig=gauss_sig, m_ang=motion_angle,m_len=motion_len,d_rad=defocus_rad)
            if noparam:
                ptensor.zero_()
            net.set_input([hrtensor,lrtensor,ptensor])
            net.validate()
            sr = net.SR
            if norm_flag is True:
                sr = utils.unnormalize_batch(sr)
            sr = sr.cpu().data.squeeze(0).clamp(-1,1)
            sr = (sr+1)/2.0 *255.0

            srIMG = tensor2img_L(sr)
            IMG = hrIMG.crop((0,0,3*w,h))
            IMG.paste(lrIMG,(w,0))
            IMG.paste(srIMG,(2*w,0))
            IMG.save(srIMGdir)

            SRPSNR = psnr(np.array(srIMG), hrnp)
            LRPSNR = psnr(lrnp, hrnp)

            addpsnr_sum += SRPSNR - LRPSNR
            psnr_sum += SRPSNR
            allpsnr_sun+=SRPSNR
            cnt +=1
            if printflag:
                print("|" + img + "\t |{:.2f}\t|{:.2f}|".format(LRPSNR, SRPSNR - LRPSNR))
    avg_psnr = psnr_sum / cnt
    avg_addpsnr = addpsnr_sum / cnt
    print('|{:3d}\t|{:.1f}({:d})\t|{:d}\t|{:d}\t|{:d}\t|{:.2f}\t|{:.2f}\t|'.format(num,gauss_sig,gauss_len,
                                                                                   motion_angle,motion_len,defocus_rad,
                                                                                   avg_addpsnr, avg_psnr))
    return allpsnr_sun,avg_psnr, avg_addpsnr

if __name__ == "__main__":
    main()