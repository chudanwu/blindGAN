
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
''
torch.cuda.set_device(0)
norm_flag = False
make_lr_flag =  False
test_unblock = False
noparams = False

refdir = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/images/banckmark/vae"
g_len_unchange = False # always=5

if g_len_unchange:
    lroutdir = os.path.join('/media/library/wcd/Datas', 'randomLR_glen5')
else:
    lroutdir = os.path.join('/media/library/wcd/Datas', 'randomLR')
opt = vaeSROption()
opt.batch_size = 1
if test_unblock:
#train by unblock
    from train_vaeSRnet import blindVAE
    opt.param_num=3
    modelname = 'come18_st2960000_block_vaeSR_gsig_3.0_glen_3_mang_89_mlen8_SRIN_block4_PBN_block8.model'
    modeldir = os.path.join("/home/wcd/Desktop/to_neural_style/ckpt/nonactivate_out/",modelname)
    if g_len_unchange:
        sroutdir = os.path.join(refdir, 'randomSR_glen5_unblock')
    else:
        sroutdir = '/media/library/wcd/Datas/randomSR_unblock'
        print('!!!!!!have not trained')
    if noparams:
        modeldir = "/home/wcd/Desktop/to_neural_style/ckpt/non_params/unblock_block4/come18_st2960000_nop_vaeSR_gsig_3.0_glen_3_mang_89_mlen8_SRIN_block4_PBN_block8.model"
        sroutdir = sroutdir + '_noparam'
else:
    from train_block_vaeSRnet import blindVAE
    print('patch training version')
    modeldir = "/media/library/wcd/Models/Train_deblur/patch_in/condition_p/come10_st960000_block_vaeSR_gsig_3.0_glen_3_mang_90_mlen10_SRIN_block16_PBN_block8.model"
    if g_len_unchange:
        sroutdir = os.path.join(refdir, 'randomSR_glen5')
    else:
        sroutdir = '/media/library/wcd/Datas/randomSR'
    if noparams:
        modelname = 'sepe18_st1440000_nop_block_vaeSR_gsig_3.0_glen_3_mang_89_mlen8_SRIN_block4_PBN_block8.model'
        modeldir = os.path.join("/home/wcd/Desktop/to_neural_style/ckpt/non_params/block4/", modelname)
        modeldir = '/home/wcd/Desktop/to_neural_style/ckpt/non_params/block4/come18_st1440000_nop_block_vaeSR_gsig_3.0_glen_3_mang_89_mlen8_SRIN_block4_PBN_block8.model'
        sroutdir = sroutdir+'_noparam'
opt.pretrain_model = modeldir

netdict = torch.load(modeldir)



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

def imgdir2tensor(img_dir):#file_name=>1x1xhxw
    img = Image.open(img_dir).convert('L')
    return transform(img).unsqueeze(0)

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


def save_tensor_image(filename, data, mode): #from cxhxw tensor
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

def tensor2img_L(tensor): #1xhxw -> hxw
    img = tensor.squeeze(0).clone().clamp(0, 255).numpy().astype("uint8")
    return Image.fromarray(img, 'L')

def main():

    test_blockn_min=1
    test_blockn_max=3
    lrpsnr=[]
    srpsnr=[]
    incpsnr = []
    # lr save dir
    if make_lr_flag:
        if os.path.exists(lroutdir):
            os.removedirs(lroutdir)
        os.mkdir(lroutdir)
        for refimg_filename in os.listdir(refdir):
            refimg_dir = os.path.join(refdir, refimg_filename)
            if not utils.is_image_file(refimg_dir):
                continue
            blur_type = np.random.randint(3)  # 0=>gauss 1=>motion
            lrimg_dir = os.path.join(lroutdir, refimg_filename)
            refimg = np.array(utils.load_HR_image(refimg_dir, size=None, mode='L'))
            h,w = refimg.shape
            test_blockn = np.random.randint(test_blockn_min, test_blockn_max + 1)
            block_str = '{:d}'.format(test_blockn)

            # 打底模糊
            if blur_type is 0:
                #gauss
                g_sig = min(np.random.rand()*opt.gauss_sig_max+0.5,opt.gauss_sig_max)
                g_len = math.ceil(g_sig*6)
                g_len = int(g_len + 1 - g_len%2)
                if g_len_unchange:
                    g_len = 5

                lrimg = utils.HRnp2LRnp(refimg,gauss_len=g_len,gauss_sig=g_sig)

                params_str = 'gauss kernel sig: {:d} {:.2f}'.format(g_len,g_sig)
                p_g_sig_img = np.full((h, w), g_sig / opt.gauss_sig_max)
            elif blur_type is 1:
                m_ang = np.random.randint(low=0,high=opt.motion_angle_max+1) # [1,max] int
                m_len = np.random.randint(low=1,high=opt.motion_len_max+1) # [0,max] int

                lrimg = utils.HRnp2LRnp(refimg,motion_angle=m_ang,motion_len=m_len)
                params_str = 'motion ang len: {:d} {:d}'.format(m_ang, m_len)
                m_x = m_len * math.cos(math.radians(m_ang))
                m_y = m_len * math.sin(math.radians(m_ang))
                p_m_x_img = np.full((h, w), m_x / opt.motion_len_max)
                p_m_y_img = np.full((h, w), m_y / opt.motion_len_max)
            else:
                d_rad = np.random.randint(low=2,high=opt.defocus_rad_max+1)
                lrimg = utils.HRnp2LRnp(refimg,defocus_rad=d_rad)
                params_str = 'defocus rad: {:d}'.format(d_rad)
                p_d_rad_img = np.full((h, w), d_rad / opt.defocus_rad_max)

            # 各blobk模糊参数
            if blur_type is 0:
                #gauss
                g_sig = np.random.rand(test_blockn)
                g_sig = (g_sig * opt.gauss_sig_max).astype(np.float32)  # [0,max) float
                g_len = np.ceil(g_sig*6)
                g_len = (g_len + 1 - g_len%2).astype(np.int)
                if g_len_unchange:
                    g_len = np.full_like(g_len,5)
            elif blur_type is 1:
                m_ang = np.random.randint(low=0,high=opt.motion_angle_max+1,size=test_blockn)# [1,max] int
                m_len = np.random.randint(low=1,high=opt.motion_len_max+1,size=test_blockn) # [0,max] int
            else:
                d_rad = np.random.randint(low=2,high=opt.defocus_rad_max+1,size=test_blockn)# [1,max] int
            h,w = refimg.shape
            for b in range(test_blockn):
                test_blocksize_min = min(8,min(w,h))
                test_blocksize_max = min(w//2,h//2)
                test_block_h = np.random.randint(test_blocksize_min, test_blocksize_max + 1, size=test_blockn)
                test_block_w = np.random.randint(test_blocksize_min, test_blocksize_max + 1, size=test_blockn)
                hstart = np.random.randint(0,h-test_block_h[b]+1)
                wstart = np.random.randint(0,w-test_block_w[b]+1)
                if blur_type is 0:
                    lrimg[hstart:hstart+test_block_h[b],wstart:wstart+test_block_w[b]] = utils.HRnp2LRnp(
                        refimg,gauss_len=g_len[b],gauss_sig=g_sig[b])[hstart:hstart+test_block_h[b],wstart:wstart+test_block_w[b]]
                    p_g_sig_img[hstart:hstart + test_block_h[b], wstart:wstart + test_block_w[b]] = np.full(
                    (test_block_h[b], test_block_w[b]), g_sig[b] / (opt.gauss_sig_max))
                    params_str = params_str+' {:d} {:.2f}'.format(g_len[b],g_sig[b])
                elif blur_type is 1:
                    lrimg[hstart:hstart + test_block_h[b], wstart:wstart + test_block_w[b]] = utils.HRnp2LRnp(
                        refimg, motion_angle=m_ang[b],motion_len=m_len[b])[hstart:hstart + test_block_h[b], wstart:wstart + test_block_w[b]]

                    m_x = m_len[b] * math.cos(math.radians(m_ang[b]))
                    m_y = m_len[b] * math.sin(math.radians(m_ang[b]))
                    #print(test_block_h[b],test_block_w[b])
                    p_m_x_img[hstart:hstart + test_block_h[b], wstart:wstart + test_block_w[b]] = np.full(
                    (test_block_h[b], test_block_w[b]), m_x / (opt.motion_len_max))
                    p_m_y_img[hstart:hstart + test_block_h[b], wstart:wstart + test_block_w[b]] = np.full(
                    (test_block_h[b], test_block_w[b]), m_y / (opt.motion_len_max))
                    params_str = params_str + ' {:d} {:d}'.format(m_ang[b], m_len[b])
                else:
                    lrimg[hstart:hstart + test_block_h[b], wstart:wstart + test_block_w[b]] = utils.HRnp2LRnp(
                        refimg, defocus_rad=d_rad[b])[hstart:hstart+test_block_h[b], wstart:wstart+test_block_w[b]]
                    p_d_rad_img[hstart:hstart + test_block_h[b], wstart:wstart + test_block_w[b]] = np.full(
                        (test_block_h[b], test_block_w[b]), d_rad[b] / (opt.defocus_rad_max))
                    params_str = params_str+' {:d}'.format(d_rad[b])
                block_str = block_str+ ' {:d} {:d} {:d} {:d}'.format(hstart,test_block_h[b],wstart,test_block_w[b])
            Image.fromarray(np.concatenate((refimg,lrimg),axis=1)).save(lrimg_dir)
            imgname = os.path.splitext(refimg_filename)
            if blur_type is 0:
                Image.fromarray((p_g_sig_img*255).astype(np.uint8)).save(os.path.join(lroutdir, imgname[0]+'_gsig'+imgname[1]))
            elif blur_type is 1:
                Image.fromarray((p_m_x_img*255).astype(np.uint8)).save(os.path.join(lroutdir, imgname[0]+'_mx'+imgname[1]))
                Image.fromarray((p_m_y_img*255).astype(np.uint8)).save(os.path.join(lroutdir, imgname[0]+'_my'+imgname[1]))
            else:
                Image.fromarray((p_d_rad_img*255).astype(np.uint8)).save(os.path.join(lroutdir, imgname[0]+'_drad'+imgname[1]))
            print(refimg_filename)
            print(block_str)
            print(params_str)
    else:
        if not os.path.exists(lroutdir):
            print('no exist path: '+lroutdir)
            return
        #读取，对比
        if not os.path.exists(sroutdir):
            os.mkdir(sroutdir)
        for refimg_filename in os.listdir(refdir):
            refimg_dir = os.path.join(refdir, refimg_filename)
            if not utils.is_image_file(refimg_dir):
                continue
            lrimg_dir = os.path.join(lroutdir, refimg_filename)
            srimg_dir = os.path.join(sroutdir, refimg_filename)
            imgname = os.path.splitext(refimg_filename)
            real_drad_dir = os.path.join(lroutdir, imgname[0] + '_drad' + imgname[1])
            fake_drad_dir = os.path.join(sroutdir, imgname[0] + '_drad' + imgname[1])
            real_gsig_dir = os.path.join(lroutdir, imgname[0] + '_gsig' + imgname[1])
            fake_gsig_dir = os.path.join(sroutdir, imgname[0] + '_gsig' + imgname[1])
            real_mx_dir = os.path.join(lroutdir, imgname[0] + '_mx' + imgname[1])
            fake_mx_dir = os.path.join(sroutdir, imgname[0] + '_mx' + imgname[1])
            real_my_dir = os.path.join(lroutdir, imgname[0] + '_my' + imgname[1])
            fake_my_dir = os.path.join(sroutdir, imgname[0] + '_my' + imgname[1])

            img = Image.open(lrimg_dir).convert('L')
            w,h = img.size
            hrimg = img.crop((0,0,w//2,h))
            lrimg = img.crop((w//2,0,w,h))
            hrtensor = transform(hrimg).unsqueeze(0)
            lrtensor = transform(lrimg).unsqueeze(0)
            if os.path.exists(real_gsig_dir):#gauss
                ptensor = torch.cat((
                    imgdir2tensor(real_gsig_dir),
                    torch.zeros(lrtensor.size()),
                    torch.zeros(lrtensor.size()),
                    torch.zeros(lrtensor.size())
                ),dim=1)
            elif os.path.exists(real_mx_dir):
                ptensor = torch.cat((
                    torch.zeros(lrtensor.size()),
                    imgdir2tensor(real_mx_dir),
                    imgdir2tensor(real_my_dir),
                    torch.zeros(lrtensor.size())
                ), dim=1)
            else:
                ptensor = torch.cat((
                    torch.zeros(lrtensor.size()),
                    torch.zeros(lrtensor.size()),
                    torch.zeros(lrtensor.size()),
                    imgdir2tensor(real_drad_dir)
                ), dim=1)
            if opt.tanhoutput:
                lrtensor = (lrtensor - 0.5) / 0.5
            if noparams:
                ptensor.zero_()
            net.set_input([hrtensor, lrtensor, ptensor])
            net.validate()
            sr = net.SR
            sr = sr.cpu().data.squeeze(0).clamp(-1, 1)
            if opt.tanhoutput:
                sr = (sr + 1) / 2.0 * 255.0
            img = img.crop((0,0,int(w*1.5),h))
            srimg = tensor2img_L(sr)
            img.paste(srimg,(w,0))
            img.save(srimg_dir)
            fake_p = net.fake_p
            fake_p = fake_p.cpu().data.clamp(0, 1)
            save_tensor_image(fake_gsig_dir, fake_p[:, 0, :, :]*255, 'L')
            save_tensor_image(fake_mx_dir, fake_p[:, 1, :, :]*255, 'L')
            save_tensor_image(fake_my_dir, fake_p[:, 2, :, :]*255, 'L')
            save_tensor_image(fake_drad_dir, (fake_p[:, 3, :, :] * 255), 'L')
            SRPSNR = psnr(np.array(srimg), np.array(hrimg))
            LRPSNR = psnr(np.array(lrimg), np.array(hrimg))
            if(LRPSNR<100):
                srpsnr.append(SRPSNR)
                lrpsnr.append(LRPSNR)
                incpsnr.append(SRPSNR-LRPSNR)
            print('| '+refimg_filename+' | {:.2f} | {:.2f} |'.format(LRPSNR,SRPSNR-LRPSNR))

        srpsnr = np.array(srpsnr)
        incpsnr = np.array(incpsnr)
        print('mean psnr:{:.2f}, std psnr:{:.2f}, mean psnr(+):{:.2f}, std psnr(+):{:.2f}'.format(srpsnr.mean(),srpsnr.std(),incpsnr.mean(),incpsnr.std()))



if __name__ == "__main__":
    main()