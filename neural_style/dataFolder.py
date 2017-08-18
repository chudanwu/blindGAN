import filters
from utils import load_HR_image,HR2LR,make_dataset,IMG_EXTENSIONS
import numpy
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
        self.imgtrans = self.tranformimg(1,crop_size=self.cropsize,norm_flag=False,mean=0.5,std=0.5)
        self.psfimgtrans = self.tranformimg(1,crop_size=None,norm_flag=False,mean=0.5,std=0.5)

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
        padkernel = numpy.pad(numpy.array(kernel), ( self.psfsize-kernel.size[0])//2, 'constant')
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
        psfnp = psfnp.astype(numpy.float32)
        psfnp /= 255
        psfnp /= psfnp.sum()
        return psfnp

    def psftotensor(self,psf):
        return torch.from_numpy(psf.astype(numpy.float32)).squeeze(0)

    def tranformimg(self,max_of_range,crop_size=None,norm_flag=False,mean=0.5,std=0.5):
        translist=[]
        if crop_size is not None:
            translist += [transforms.CenterCrop(crop_size)]
        translist += [transforms.ToTensor()]
        if max_of_range is not 1:
            translist += [transforms.Lambda(lambda x_HR: x_HR.mul(max_of_range))]
        if norm_flag:
            if self.mode is 'L':
                translist += [transforms.Normalize(mean,std)]
            elif self.mode is 'RGB':
                translist += [transforms.Normalize((mean,mean,mean),(std,std,std))]
        return transforms.Compose(translist)



    def __len__(self):
        return len(self.imgs)//3