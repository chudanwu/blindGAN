import os
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms

import utils
import filters

torch.cuda.set_device(1)
norm_flag = False
make_lr_flag = False

refdir = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/images/style-images"
lrdir = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/images/style-images/SR"

modelname = 'SRnet_L_len0_ang33_g1.7_unorm_US_l1.model'
modeldir = os.path.join("/home/wcd/Projects/Pytorch-examples/fast_neural_style/neural_style/ckpt",modelname)

lrnetdict = torch.load(modeldir)
opt = lrnetdict["args"]

lrnet = lrnetdict["net"]
lrnet.load_state_dict(lrnetdict["state_dict"])
lrnet.cuda()


motion_len = opt.motion_len

motion_angel = opt.motion_angel

gauss = opt.gauss

motion_kernel,motion_anchor = filters.motion_kernel(motion_len,motion_angel)

mode = opt.mode

argstr="len_"+str(motion_len)+"ang_"+str(motion_angel)+"g_"+str(gauss)

outdir = os.path.join(refdir,argstr)
if not os.path.exists(outdir):
    os.mkdir(outdir)
    os.mkdir(os.path.join(os.path.join(outdir,"LR")))
    os.mkdir(os.path.join(os.path.join(outdir,"SR")))


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

def psnr(im1,im2):
    MSE = mse(im1,im2)
    psnr = 20*np.log10(255)-10*np.log10(MSE)
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

print("save dir: "+outdir)

print("|img-name \t|blurPSNR | PSNR |")

for img in os.listdir(refdir):
    hrIMGdir = os.path.join(refdir, img)
    if make_lr_flag is True:
        lrIMGdir = os.path.join(os.path.join(outdir,"LR"), img)
    else:
        lrIMGdir = os.path.join(lrdir, img)
    srIMGdir = os.path.join(os.path.join(outdir,"SR"), img)


    if utils.is_image_file(img):
        hrIMG = utils.load_HR_image(hrIMGdir,size=None,mode=mode)
        lrIMG = utils.HR2LR(hrIMG,motion_kernel,motion_anchor,gauss)
        if make_lr_flag is True:
            lrIMG.save(lrIMGdir)
        lr = img2Variable(lrIMG).cuda()

        if norm_flag is True:
            lr = utils.normalize_batch(lr)
        sr = lrnet(lr)
        if norm_flag is True:
            sr = utils.unnormalize_batch(sr)
        sr = sr.cpu().data.squeeze(0)
        save_image(srIMGdir,sr,mode)
        SRPSNR = psnr(sr.squeeze(0).clamp(0,255).numpy(),np.array(hrIMG))
        LRPSNR = psnr(lr.data.cpu().squeeze(0).squeeze(0).clamp(0,255).numpy(), np.array(hrIMG))
        print("|"+img+"\t |{:.6f}\t|{:.6f}|".format(LRPSNR,SRPSNR-LRPSNR))

