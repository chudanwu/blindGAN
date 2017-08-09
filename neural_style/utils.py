import torch
from PIL import Image,ImageFilter
from torch.autograd import Variable
import torch.utils.data as data
import os
import os.path
import filters
import numpy
import time

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# load img from floder using filename
def load_HR_image(filename, size=None, mode='RGB'):
    # input: img path
    # output: PIL img
    if mode is 'Y':
        y, cb, cr = Image.open(filename).convert('YCbCr').split()
        img = y
    elif mode is 'L':
        img =  Image.open(filename).convert('L')
    elif mode is 'RGB':
        img = Image.open(filename).convert('RGB')
    elif mode is 'YCbCr':
        img = Image.open(filename).convert('YCbCr')

    h, w = img.size
    img = img.crop((0, 0, h - h % 4, w - w % 4))

    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    return img


def load_LR_image(filename, motion_len=0,motion_angel=0,gauss=0,size=None,scale=None, mode='Y'):
    if mode is 'Y':
        y, cb, cr = Image.open(filename).convert('YCbCr').split()
        img = y
    elif mode is 'L':
        img =  Image.open(filename).convert('L')
    elif mode is 'RGB':
        img = Image.open(filename).convert('RGB')
    elif mode is 'YCbCr':
        img = Image.open(filename).convert('YCbCr')

    h, w = img.size
    img = img.crop((0, 0, h - h % 4, w - w % 4))

    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(size / scale), int(size / scale)), Image.ANTIALIAS)


    if motion_len is not 0 and motion_angel is not 0:
        motion_kernel,motion_anchor = filters.motion_kernel(motion_len,motion_angel)
        img = filters.motion_blur(img,motion_kernel,motion_anchor)
    if gauss is not 0:
        img = filters.gauss_blur(img,gauss)

    return img


# Load content img
def default_loader(path):
    l = Image.open(path).convert('L')
    return Image.merge("RGB", (l, l, l))


# transfer HR PIL img to LR PIL img
def HR2LR(img,motion_kernel=None,motion_anchor=None,gauss=None,scale=None):
    if motion_kernel is not None and motion_anchor is not None:
        img = filters.motion_blur(img,motion_kernel,motion_anchor)
    if gauss is not 0:
        img = filters.gauss_blur(img,gauss)
    if scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


class trainingFolder(data.Dataset):

    def __init__(self, root, HR_size, LR_scale=None,
                 motion_len=None,motion_angel=None,gauss=None,
                 transform=None, target_transform=None,loader=load_HR_image,mode='Y'):
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
        #training img--content
        img = self.loader(path,self.HR_size,mode=self.mode)
        #target img--reference
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
        blur = self.loader(imgdir+"_blur.png",self.HR_size,mode=self.mode)
        #target img--reference
        orig = self.loader(imgdir+"_orig.png",self.HR_size,mode=self.mode)
        psf = self.psfloader(imgdir+"_psf.png")

        if self.transform is not None:
            blur = self.transform(blur)
            orig = self.transform(orig)
            psf = self.transform(psf)

        return psf,orig,blur

    def psfloader(self,filename):
        maxsize = 29
        img = Image.open(filename).convert('L')
        padimg = numpy.pad(numpy.array(img), (maxsize-img.size[0])//2, 'constant')
        return Image.fromarray(padimg,'L')


    def __len__(self):
        return len(self.imgs)//3

def make_dataset(dir):
    images = []
    for target in os.listdir(dir):
        fname = os.path.join(dir, target)
        if is_image_file(fname):
            images.append(fname)
    return images


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.div(batch, 255.0)
    batch -= Variable(mean, requires_grad=False)
    batch /= Variable(std, requires_grad=False)
    return batch

def unnormalize_batch(batch):
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.mul(batch,Variable(std,requires_grad=False))
    batch += Variable(mean,requires_grad=False)
    batch = torch.mul(batch, 255.0)
    return batch

def save_model(net,args,val_loss,end_epoch,batch_id):
    if args.normalizebatch_flag is True:
        normflag='norm'
    else:
        normflag='unorm'
    model_filename = args.model_name+ "_" +args.mode +"_len"+str(args.motion_len)+"_ang"+str(args.motion_angel)+"_g"+\
                     str(args.gauss)+"_"+normflag+"_"+args.deconv_mode+"_"+args.lossfun+".model"
    model_path = os.path.join(args.save_model_dir, model_filename)
    model_dict= {"state_dict":net.state_dict(),
                 "net":net,
                 "args":args,
                 "loss_function":args.lossfun,
                 "end_epoch":end_epoch,
                 "val_loss":val_loss,
                 "batch_id":batch_id}
    torch.save(model_dict, model_path)

# 保存和加载整个模型:
# torch.save(model_object, 'model.pkl')
# model = torch.load('model.pkl')
# 仅保存和加载模型参数(推荐使用):
# torch.save(model_object.state_dict(), 'params.pkl')
# model_object.load_state_dict(torch.load('params.pkl'))

def load_model(net,model_path,return_modeldict=False):
    model_dict = torch.load(model_path)
    net.load_state_dict(model_dict["state_dict"])
    if return_modeldict is True:
        return model_dict

# now we can creat net object accroding to different color_mode settings

class TVLoss(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()

    def forward(self,x):
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return h_tv/count_h + w_tv/count_w

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class L1Reg(torch.nn.Module):
    def __init__(self):
        super(L1Reg,self).__init__()

    def forward(self, x):
        xsize = self._tensor_size(x)
        return x.sum()/xsize

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class L1edgeReg(torch.nn.Module):
    def __init__(self):
        super(L1edgeReg, self).__init__()

    def forward(self, x):
        h_x = x.size()[2]
        w_x = x.size()[3]
        hd = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2)
        wd = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2)
        grad = torch.pow(torch.add(hd,wd),0.5)
        gradsize = self._tensor_size(grad)
        return grad.sum()/gradsize

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]