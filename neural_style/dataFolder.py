import filters
from utils import load_HR_image,HR2LR,make_dataset,IMG_EXTENSIONS,HRnp2LRnp
import numpy as np
import math
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

class trainingFolder(data.Dataset):

    def __init__(self, root, HR_size, LR_scale=None,
                 motion_len=None,motion_angel=None,gauss=None,
                 transform=None, target_transform=None, loader=load_HR_image,mode='Y'):
        super(trainingFolder,self).__init__()
        self.loader = loader
        if motion_len is not None and motion_angel is not None:
            self.motion_kernel, self.motion_anchor = filters.motion_kernel(motion_len, motion_angel)
        self.gauss = gauss
        self.HR_size = HR_size
        self.LR_scale = LR_scale
        self.mode=mode
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = make_dataset(root)
        if len(self.imgs ) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
    def __getitem__(self, index):
        path = self.imgs[index]
        # training img--content
        img = self.loader(path,self.HR_size,mode=self.mode)
        # target img--reference
        target = HR2LR(img,self.motion_kernel,self.motion_anchor,self.gauss,self.LR_scale)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

class randomBlurFolder(data.Dataset):
    '''
    random LR-HR-param pair,(parameter of LR is random)
    '''
    def __init__(self, root, HR_size, LR_scale=None,
                 motion_len_max=None,motion_angel_max=360,gauss_max=None,
                 transform=None, target_transform=None, loader=load_HR_image,mode='L'):
        super(randomBlurFolder,self).__init__()
        self.loader = loader
        self.motion_len_max = motion_len_max
        self.motion_angel_max = motion_angel_max
        self.gauss_max = gauss_max
        self.HR_size = HR_size
        self.LR_scale = LR_scale
        self.mode=mode
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = make_dataset(root)
        if len(self.imgs ) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
    def __getitem__(self, index):
        path = self.imgs[index]
        # training img--content
        HR = self.loader(path,self.HR_size,mode=self.mode)
        blur_type = np.random.randint(2) #gauss[0] or motion[1] or gauss+motion[2]
        # target img--reference
        if self.gauss_max >= 0 and blur_type is not 1:
            gauss = np.random.rand()*self.gauss_max
        else:
            gauss = 0
        if self.motion_len_max >0 and blur_type is not 0:
            motion_len = np.random.randint(self.motion_len_max-1)+1
            motion_angel = np.random.randint(self.motion_angel_max) #-0.5*self.motion_angel_max
            motion_kernel, motion_anchor = filters.motion_kernel(motion_len, motion_angel)
            motion_x = motion_len*math.cos(math.radians(motion_angel))
            motion_y = motion_len*math.sin(math.radians(motion_angel))
        else:
            motion_kernel = None
            motion_anchor =None
            motion_x = 0
            motion_y = 0
        LR = HR2LR(HR,motion_kernel,motion_anchor,gauss)

        if self.transform is not None:
            HR = self.transform(HR)
        if self.target_transform is not None:
            LR = self.target_transform(LR)
        params = torch.from_numpy(np.array([gauss,motion_x,motion_y]).astype(np.float32))
        #print([gauss,motion_x,motion_y])
        return HR, LR, params

    def __len__(self):
        return len(self.imgs)

class vaeSRFolder(data.Dataset):
    '''
    random LR-HR-param pair,(parameter of LR is random)
    '''
    def __init__(self, opt):
        super(vaeSRFolder,self).__init__()
        self.gauss_len_max = opt.gauss_len_max # 3 (3 5 7
        self.gauss_sig_max = opt.gauss_sig_max # 3.0 (0-3.0
        self.motion_len_max = opt.motion_len_max  # 8 (0-8
        self.motion_angle_max = opt.motion_angle_max # 89 (1-89
        self.img_size = opt.img_size
        self.param_num = opt.param_num
        self.mode= opt.color_mode
        self.tanhoutput = opt.tanhoutput
        self.imgtrans = self.tranformimg(max_of_range=1,mean=0.5,std=0.5)
        self.imgs = make_dataset(opt.train_dir)
        print('random blur (consistance) training data')
        if len(self.imgs ) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + opt.train_dir + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
    def __getitem__(self, index):
        path = self.imgs[index]
        # training img--content
        HR = load_HR_image(path,mode=self.mode,centercrop=self.img_size)
        params = np.zeros((self.param_num, self.img_size, self.img_size), dtype=np.float32)
        blur_type = np.random.randint(2) #gauss[0] or motion[1] or gauss+motion[2]
        # target img--reference
        if not blur_type:
            #gauss
            gauss_len = np.random.randint(low=1, high=self.gauss_len_max + 1) * 2 + 1  # [3,2max+1] int
            gauss_sig = (np.random.rand() * self.gauss_sig_max) # [0,max) float
            motion_angle = 0
            motion_len = 0
            params[0] = np.full((self.img_size,self.img_size),gauss_len/(2 * self.gauss_len_max + 1))
            params[1] = np.full((self.img_size,self.img_size),gauss_sig/ self.gauss_sig_max )
        else:
            #motion
            motion_angle = np.random.randint(low=1, high=self.motion_angle_max+1)  # [1,max] int
            motion_len = np.random.randint(low=0, high=self.motion_len_max+1)  # [0,max] int
            motion_kernel, motion_anchor = filters.motion_kernel(motion_len, motion_angle)
            motion_x = motion_len*math.cos(math.radians(motion_angle))
            motion_y = motion_len*math.sin(math.radians(motion_angle))
            gauss_len = 0
            gauss_sig = 0
            params[2] = np.full((self.img_size,self.img_size),motion_x/(self.motion_len_max))
            params[3] = np.full((self.img_size,self.img_size),motion_y/(self.motion_len_max))


        Img = np.array(HR)
        Img = HRnp2LRnp(Img, gauss_len, gauss_sig, motion_angle, motion_len)

        if self.imgtrans is not None:
            HR = self.imgtrans(HR)
            LR = self.imgtrans(Image.fromarray(Img))
        #print([gauss,motion_x,motion_y])
        params = torch.from_numpy(params).contiguous()
        return HR, LR, params

    def __len__(self):
        return len(self.imgs)

    def tranformimg(self,max_of_range=1,crop_size=None,mean=0.5,std=0.5):
        translist=[]
        if crop_size is not None:
            translist += [transforms.CenterCrop(crop_size)]
        translist += [transforms.ToTensor()]
        if self.tanhoutput:
            if self.mode is 'RGB':
                translist += [transforms.Normalize(mean,std)]
            elif self.mode is 'L':
                translist += [transforms.Normalize((mean,mean,mean),(std,std,std))]
        if max_of_range is not 1:
            translist += [transforms.Lambda(lambda x_HR: x_HR.mul(max_of_range))]
        return transforms.Compose(translist)

class block_vaeSRFolder(data.Dataset):
    '''
    random LR-HR-param pair,(parameter of LR is random)
    '''
    def __init__(self, opt):
        super(block_vaeSRFolder,self).__init__()
        self.gauss_len_max = opt.gauss_len_max
        self.gauss_sig_max = opt.gauss_sig_max
        self.motion_angle_max = opt.motion_angle_max #89
        self.motion_len_max = opt.motion_len_max #8
        self.block_scale = opt.block_scale
        self.img_size = opt.img_size
        self.mode = opt.color_mode
        self.tanhoutput = opt.tanhoutput
        self.param_num = opt.param_num
        self.imgtrans = self.tranformimg(max_of_range=1,mean=0.5,std=0.5)
        self.imgs = make_dataset(opt.train_dir)
        print('random block blur (unconsistance) training data')
        if len(self.imgs ) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + opt.train_dir + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
    def __getitem__(self, index):
        path = self.imgs[index]
        # training img--content
        HR = load_HR_image(path,mode=self.mode,centercrop=self.img_size)
        blur_type = np.random.randint(2) #gauss[0] or motion[1]
        scale = np.random.randint(self.block_scale)
        block_size = self.img_size // (2 ** scale) #img_size=128/256/512
        block_num = 2 ** (2 * scale)
        # target img--reference
        if not blur_type:
            #gauss
            gauss_sig = np.random.rand(block_num)
            gauss_sig = (gauss_sig * self.gauss_sig_max).astype(np.float32)  # [0,max) float
            gauss_len = np.random.randint(low=1, high=self.gauss_len_max + 1, size=block_num) * 2 + 1  # [3,2max+1] int
        else:
            motion_angle = np.random.randint(low=1,high=self.motion_angle_max +1 ,size=block_num) # [1,89] int
            motion_len = np.random.randint(low=0,high=self.motion_len_max +1 ,size=block_num) # [0,9] int

        Img = np.array(HR)
        newImg = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        params = np.zeros((self.param_num, self.img_size, self.img_size), dtype=np.float32)  # len,sig, x,y

        for b in range(block_num):
            row = b // 2 ** scale
            col = b % 2 ** scale
            id_x = col * block_size
            id_y = row * block_size
            if not blur_type:
                newImg[id_y:id_y + block_size, id_x:id_x + block_size] = HRnp2LRnp(Img, gauss_len[b],
                                                                                             gauss_sig[b], 0, 0)[
                                                                                     id_y:id_y + block_size,
                                                                                     id_x:id_x + block_size]
                params[0, id_y:id_y + block_size, id_x:id_x + block_size] = np.full(
                    (block_size, block_size), gauss_len[b] / (2 * self.gauss_len_max + 1))
                params[1, id_y:id_y + block_size, id_x:id_x + block_size] = np.full(
                    (block_size, block_size), gauss_sig[b] / (self.gauss_sig_max))
            else:
                newImg[id_y:id_y + block_size, id_x:id_x + block_size] = HRnp2LRnp(Img, 0, 0,
                                                                                             motion_angle[b],
                                                                                             motion_len[b])[
                                                                                     id_y:id_y + block_size,
                                                                                     id_x:id_x + block_size]
                m_x = motion_len[b] * math.cos(math.radians(motion_angle[b]))
                m_y = motion_len[b] * math.sin(math.radians(motion_angle[b]))
                params[2, id_y:id_y + block_size, id_x:id_x + block_size] = np.full(
                    (block_size, block_size), m_x / (self.motion_len_max)) #  m_x/8
                params[3, id_y:id_y + block_size, id_x:id_x + block_size] = np.full(
                    (block_size, block_size), m_y / (self.motion_len_max)) # m_y/8
        LR = Image.fromarray(newImg)
        if self.imgtrans is not None:
            HR = self.imgtrans(HR)
            LR = self.imgtrans(LR)
        params = torch.from_numpy(params).contiguous()
        #print([gauss,motion_x,motion_y])
        return HR, LR, params

    def __len__(self):
        return len(self.imgs)

    def tranformimg(self,max_of_range=1,crop_size=None,mean=0.5,std=0.5):
        translist=[]
        translist += [transforms.ToTensor()]
        if self.tanhoutput:
            if self.mode is 'RGB':
                translist += [transforms.Normalize(mean,std)]
            elif self.mode is 'L':
                translist += [transforms.Normalize((mean,mean,mean),(std,std,std))]
        if max_of_range is not 1:
            translist += [transforms.Lambda(lambda x_HR: x_HR.mul(max_of_range))]
        return transforms.Compose(translist)

class BMVCFolder(data.Dataset):

    def __init__(self, root, HR_size, LR_scale=None,
                 transform=None, target_transform=None,loader=load_HR_image,mode='L'):
        super(BMVCFolder,self).__init__()
        self.loader = loader

        self.HR_size = HR_size
        self.LR_scale = LR_scale
        self.mode=mode
        self.transform = transform
        self.target_transform = target_transform
        self.dir = root
        self.imgs = make_dataset(root)
        if len(self.imgs ) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
    def __getitem__(self, index):
        imgdir = self.dir + "/" + str(index).zfill(7)
        #training img--content
        img = self.loader(imgdir+"_blur.png",self.HR_size,mode=self.mode)
        #target img--reference
        target = self.loader(imgdir+"_orig.png",self.HR_size,mode=self.mode)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)//3

class BMVC_blur_psf_orig_Folder(data.Dataset):

    def __init__(self, opt):
        super(BMVC_blur_psf_orig_Folder,self).__init__()

        # color mode
        self.mode=opt.color_mode
        # to pad psf into certain size
        self.psfsize = opt.psf_size
        # to crop img into certain size
        self.cropsize = opt.img_size

        # transform function for -> training tensor
        self.imgtrans = self.tranformimg(1,crop_size=self.cropsize,norm_flag=True)
        self.psfimgtrans = self.tranformimg(1,crop_size=None,norm_flag=True)

        # img database info: path、name
        self.dir = opt.train_dir
        self.imgs = make_dataset(self.dir)

        if len(self.imgs ) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + self.dir + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

    def __getitem__(self, index):

        imgdir = self.dir + "/" + str(index).zfill(7)

        orig = self.load_orig(imgdir+"_orig.png",scalesize=None)  # PIL Image,0-255 hxwxc
        psf = self.load_psf(imgdir+"_psf.png")  # np 0-255 29x29x1
        normpsfnp = self.norm_psf(psf) # np 0-verysmall 29x29x1
        blur = filters.psf_blur(orig,normpsfnp)  # PIL Image,0-255 hxwxc

        blur = self.imgtrans(blur)  # tensor 0-1 cxhxw
        orig = self.imgtrans(orig)  # tensor 0-1 cxhxw
        unnormpsf = self.psfimgtrans(Image.fromarray(psf))  # tensor 0-1,1x29x29
        #normpsf = self.psftotensor(normpsfnp)  # very small float tensor used to conv, 1x29x29

        return blur,unnormpsf,orig

    def load_psf(self,filename):

        # input: img path
        # oputput: padded np kernel 0-255
        kernel = Image.open(filename).convert('L')
        padkernel = np.pad(np.array(kernel), ( self.psfsize-kernel.size[0])//2, 'constant')
        return padkernel

    def load_orig(self,filename, scalesize=None):

        # input: img path
        # output: PIL img
        if self.mode is 'L':
            img = Image.open(filename).convert('L')
        elif self.mode is 'RGB':
            img = Image.open(filename).convert('RGB')

        h, w = img.size
        img = img.crop((0, 0, h - h % 4, w - w % 4))

        if scalesize is not None:
            img = img.resize((scalesize, scalesize), Image.ANTIALIAS)
        return img

    def norm_psf(self,psfnp):
        psfnp = psfnp.astype(np.float32)
        psfnp /= 255
        psfnp /= psfnp.sum()
        return psfnp

    def psftotensor(self,psf):
        return torch.from_numpy(psf.astype(np.float32)).squeeze(0)

    def tranformimg(self,max_of_range,crop_size=None,norm_flag=False,mean=0.5,std=0.5):
        translist=[]
        if crop_size is not None:
            translist += [transforms.CenterCrop(crop_size)]
        translist += [transforms.ToTensor()]
        if norm_flag:
            if self.mode is 'RGB':
                translist += [transforms.Normalize(mean,std)]
            elif self.mode is 'L':
                translist += [transforms.Normalize((mean,mean,mean),(std,std,std))]
        if max_of_range is not 1:
            translist += [transforms.Lambda(lambda x_HR: x_HR.mul(max_of_range))]
        return transforms.Compose(translist)



    def __len__(self):
        return len(self.imgs)//3

import os
class patch_matlab3blur_Folder(data.Dataset):
    '''
    random LR-HR-param pair,(parameter of LR is random)
    '''
    def __init__(self, opt,max_num=50000):
        super(patch_matlab3blur_Folder,self).__init__()
        self.gauss_sig_max = opt.gauss_sig_max
        self.save_train_img = opt.save_train_img  # true
        if opt.save_train_img :
            self.save_train_img_dir = opt.save_train_img_dir
            self.save_count = [0,0,0]
        self.motion_angle_max = opt.motion_angle_max #89
        self.motion_len_max = opt.motion_len_max #8
        self.defocus_rad_max = opt.defocus_rad_max
        self.block_scale = opt.block_scale
        self.img_size = opt.img_size
        self.mode = opt.color_mode
        self.tanhoutput = opt.tanhoutput
        self.blur_type_str = ['g','m','d']
        self.param_num = opt.param_num # 4-> gauss_sig, motion_x,motion_y, defocus_len
        assert self.param_num == 4
        self.imgtrans = self.tranformimg(max_of_range=1,mean=0.5,std=0.5)
        self.imgs = make_dataset(opt.train_dir)
        self.max_num=max_num
        print('random block blur (unconsistance) training data')
        if len(self.imgs ) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + opt.train_dir + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
    def __getitem__(self, index):
        path = self.imgs[index]
        # training img--content
        HR = load_HR_image(path,mode=self.mode,centercrop=self.img_size)
        blur_type = np.random.randint(3) #gauss[0] or motion[1] or defocus[2]
        scale = np.random.randint(self.block_scale)
        block_size = self.img_size // (2 ** scale) #img_size=128/256/512
        block_num = 2 ** (2 * scale)
        # target img--reference
        if blur_type is 0:
            #gauss
            gauss_sig = np.random.rand(block_num)
            gauss_sig = (gauss_sig * self.gauss_sig_max).astype(np.float32)  # [0,max) float
            gauss_len = np.ceil(gauss_sig*6)
            gauss_len = (gauss_len + 1-gauss_len%2).astype(np.int)
        elif blur_type is 1:
            motion_angle = np.random.randint(low=0,high=self.motion_angle_max +1 ,size=block_num) # [0,max] int
            motion_len = np.random.randint(low=1,high=self.motion_len_max +1 ,size=block_num) # [1,max] int
        else :
            defocus_rad = np.random.randint(low=2,high=self.defocus_rad_max+1, size=block_num) # [2,max]

        Img = np.array(HR)
        newImg = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        params = np.zeros((self.param_num, self.img_size, self.img_size), dtype=np.float32)  # sig, x,y, rad

        for b in range(block_num):
            row = b // 2 ** scale
            col = b % 2 ** scale
            id_x = col * block_size
            id_x_end = id_x + block_size
            id_y = row * block_size
            id_y_end = id_y + block_size
            if blur_type is 0: # gauss
                newImg[id_y:id_y_end,id_x:id_x_end] = HRnp2LRnp(Img, gauss_len=gauss_len[b],gauss_sig = gauss_sig[b])[
                                                      id_y:id_y_end,id_x:id_x_end]
                #params[0, id_y:id_y + block_size, id_x:id_x + block_size] = np.full(
                #    (block_size, block_size), gauss_len[b] / (2 * self.gauss_len_max + 1))
                params[0, id_y:id_y + block_size, id_x:id_x + block_size] = np.full(
                    (block_size, block_size), gauss_sig[b] / (self.gauss_sig_max))
            elif blur_type is 1: # motion
                newImg[id_y:id_y + block_size,id_x:id_x_end] = HRnp2LRnp(Img,motion_angle=motion_angle[b],motion_len=motion_len[b])[
                                                               id_y:id_y + block_size,id_x:id_x_end]
                m_x = motion_len[b] * math.cos(math.radians(motion_angle[b]))
                m_y = motion_len[b] * math.sin(math.radians(motion_angle[b]))
                params[1, id_y:id_y + block_size, id_x:id_x + block_size] = np.full(
                    (block_size, block_size), m_x / (self.motion_len_max)) #  m_x/8
                params[2, id_y:id_y + block_size, id_x:id_x + block_size] = np.full(
                    (block_size, block_size), m_y / (self.motion_len_max)) # m_y/8
            else: # defocus
                newImg[id_y:id_y + block_size,id_x:id_x_end] = HRnp2LRnp(Img,defocus_rad=defocus_rad[b])[
                                                               id_y:id_y + block_size,id_x:id_x_end]
                params[3, id_y:id_y + block_size, id_x:id_x + block_size] = np.full(
                    (block_size, block_size), defocus_rad[b] / (self.defocus_rad_max))
        LR = Image.fromarray(newImg)
        if self.save_train_img:
            saveImg = Image.fromarray(np.concatenate((HR,newImg),axis=1))
            saveName = self.blur_type_str[blur_type] + '_{:012d}.png'.format(self.save_count[blur_type])
            self.save_count[blur_type] += 1
            saveImg.save(os.path.join(self.save_train_img_dir,saveName))
        if self.imgtrans is not None:
            HR = self.imgtrans(HR)
            LR = self.imgtrans(LR)
        params = torch.from_numpy(params).contiguous()
        #print([gauss,motion_x,motion_y])
        return HR, LR, params

    def __len__(self):
        return min(self.max_num,len(self.imgs))

    def tranformimg(self,max_of_range=1,crop_size=None,mean=0.5,std=0.5):
        translist=[]
        translist += [transforms.ToTensor()]
        if self.tanhoutput:
            if self.mode is 'RGB':
                translist += [transforms.Normalize(mean,std)]
            elif self.mode is 'L':
                translist += [transforms.Normalize((mean,mean,mean),(std,std,std))]
        if max_of_range is not 1:
            translist += [transforms.Lambda(lambda x_HR: x_HR.mul(max_of_range))]
        return transforms.Compose(translist)

class matlab3blur_Folder(data.Dataset):
    '''
    random LR-HR-param pair,(parameter of LR is random)
    '''
    def __init__(self, opt):
        super(matlab3blur_Folder,self).__init__()
        self.defocus_rad_max = opt.defocus_rad_max # 22 (2-22
        self.gauss_sig_max = opt.gauss_sig_max # 5.0 (0-5.0
        self.motion_len_max = opt.motion_len_max  # 8 (0-8
        self.motion_angle_max = opt.motion_angle_max # 89 (1-89
        self.img_size = opt.img_size
        self.param_num = opt.param_num
        self.mode= opt.color_mode
        self.tanhoutput = opt.tanhoutput
        self.imgtrans = self.tranformimg(max_of_range=1,mean=0.5,std=0.5)
        self.imgs = make_dataset(opt.train_dir)
        print('random blur (consistance) training data')
        if len(self.imgs ) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + opt.train_dir + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
    def __getitem__(self, index):
        path = self.imgs[index]
        # training img--content
        HR = load_HR_image(path,mode=self.mode,centercrop=self.img_size)
        params = np.zeros((self.param_num, self.img_size, self.img_size), dtype=np.float32)
        blur_type = np.random.randint(2) #gauss[0] or motion[1] or gauss+motion[2]
        # target img--reference
        gauss_len = 0
        gauss_sig = 0
        motion_angle = 0
        motion_len = 0
        defocus_rad = 0
        if blur_type is 0:
            #gauss
            gauss_sig = (np.random.rand() * self.gauss_sig_max) # [0,max) float
            gauss_len = math.ceil(gauss_sig*6)
            gauss_len = int(gauss_len + 1 - gauss_len%2)
            params[0] = np.full((self.img_size,self.img_size),gauss_sig/ self.gauss_sig_max )
        elif blur_type is 1:
            #motion
            motion_angle = np.random.randint(low=1, high=self.motion_angle_max+1)  # [1,max] int
            motion_len = np.random.randint(low=0, high=self.motion_len_max+1)  # [0,max] int
            #motion_kernel, motion_anchor = filters.motion_kernel(motion_len, motion_angle)
            motion_x = motion_len*math.cos(math.radians(motion_angle))
            motion_y = motion_len*math.sin(math.radians(motion_angle))
            params[1] = np.full((self.img_size,self.img_size),motion_x/(self.motion_len_max))
            params[2] = np.full((self.img_size,self.img_size),motion_y/(self.motion_len_max))
        else:
            #defocus
            defocus_rad = np.random.randint(low=2, high=self.defocus_rad_max+1) # [2,22] int
            params[3] = np.full((self.img_size,self.img_size),defocus_rad/(self.defocus_rad_max))


        Img = np.array(HR)
        Img = HRnp2LRnp(Img, gauss_len=gauss_len, gauss_sig=gauss_sig, motion_angle=motion_angle,
                        motion_len=motion_len,defocus_rad=defocus_rad)

        if self.imgtrans is not None:
            HR = self.imgtrans(HR)
            LR = self.imgtrans(Image.fromarray(Img))
        #print([gauss,motion_x,motion_y])
        params = torch.from_numpy(params).contiguous()
        return HR, LR, params
