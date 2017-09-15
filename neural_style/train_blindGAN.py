
import time
import os
import torch.nn.functional as F
import torch
import  torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import print_params_num, ImagePool, tensor2im, tensor2im_tanh
import draw
from dataFolder import BMVC_blur_psf_orig_Folder
import models
from options import Option

loss_dict = {'mse':nn.MSELoss, 'l1':nn.L1Loss, 'bce':nn.BCELoss}
channel_of_colormode = {'L':1,'Y':1,'RGB':3}

# blindGAN trainer
class blindGAN():
    def name(self):
        return 'blindGAN'

    def __init__(self, opt):

        self.isTrain = True
        self.opt = opt
        if self.opt.cudaid is not None:
            torch.cuda.set_device(self.opt.cudaid)

        # define tensors in/not in cuda
        self.Tensor = torch.cuda.FloatTensor if opt.cudaid else torch.FloatTensor
        self.input_blur = self.Tensor(opt.batch_size,channel_of_colormode[opt.color_mode],
                                   opt.img_size,opt.img_size)
        self.input_orig = self.input_blur.clone()
        self.input_psf = self.Tensor(opt.batch_size, 1,opt.psf_size, opt.psf_size)
        self.input_id_psf = self.input_psf.clone()
        self.set_identity_psf(self.input_id_psf)

        # load/define networks
        self.netG = models.GpsfNet(mode=opt.color_mode, out_c=1,resblock_num=opt.G_block_num,
                                   downscale=opt.G_bottle_scale,norm_mode=opt.G_norm_mode,drop_out=opt.dropout)
        if self.opt.cudaid is not 0:
            self.netG = self.netG.cuda()

        if not self.isTrain : # or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            use_lsgan = True if opt.ganloss is 'mse' else False
            use_sigmoid = not use_lsgan
            self.netD = models.DNet(opt.img_by_psf**2+1, insize=opt.psf_size, out_c=1, n_layers=opt.D_layer_nun,
                                    norm_mode=opt.D_norm_mode, use_sigmoid=use_sigmoid)
            if self.opt.cudaid is not 0:
                self.netD = self.netD.cuda()
            self.fake_psf_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = models.GANLoss(use_lsgan=use_lsgan, tensor=self.Tensor)
            self.criterionGenerate = loss_dict[opt.generate_loss]()
            self.criterionID = loss_dict[opt.identity_loss]()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.optimize_mode is not 'admm':
                raise ValueError('optimize_mode: %s undifine'%(opt.optimize_mode))

        print('---------- Networks initialized -------------')
        print_params_num(self.netG)
        if self.isTrain:
            print_params_num(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input): #blur,unnormpsf(0-1),orig
        input_blur = input[0]
        input_psf = input[1]
        input_orig = input[2]
        self.input_blur.resize_(input_blur.size()).copy_(input_blur)
        self.input_psf.resize_(input_psf.size()).copy_(input_psf)
        self.input_orig.resize_(input_orig.size()).copy_(input_orig)

    def forward(self):
        self.real_blur = Variable(self.input_blur)
        self.fake_psf = self.netG(self.real_blur)
        self.real_psf = Variable(self.input_psf)
        self.id_psf = Variable(self.input_id_psf)
        self.orig = Variable(self.input_orig)
        self.fake_id_psf = self.netG(self.orig)

    # no backprop gradients
    def test(self):
        self.real_blur = Variable(self.input_blur, volatile=True)
        self.fake_psf = self.netG(self.real_blur)
        self.real_psf = Variable(self.input_psf, volatile=True)

    def backward_D(self): #forward+backward

        # the condition input of D
        real_blur = self.real_blur.view(self.opt.batch_size,-1,self.opt.psf_size,self.opt.psf_size)
        # Fake
        # stop backprop to the generator by detaching fake_psf
        fake_blurpsf = self.fake_psf_pool.query(torch.cat((real_blur, self.fake_psf), 1))
        self.pred_fake = self.netD(fake_blurpsf.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_blurpsf = torch.cat((real_blur, self.real_psf), 1)
        self.pred_real = self.netD(real_blurpsf)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward() #GAN loss but stop at D

    def backward_G(self):

        # the condition input of D
        real_blur = self.real_blur.view(self.opt.batch_size,-1,self.opt.psf_size,self.opt.psf_size)

        # First, G() should fake the discriminator
        fake_blurpsf = torch.cat((real_blur, self.fake_psf), 1)
        pred_fake = self.netD(fake_blurpsf)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(blur) = psf
        self.loss_G_gen = self.criterionGenerate(self.fake_psf, self.real_psf) * self.opt.lambda1

        # Third, G(orig) = id_psf
        self.loss_G_id = self.criterionID(self.fake_id_psf,self.id_psf) * self.opt.lambda2

        # Fourth, orig * G(blur)_norm = blur
        fake_psf_norm = self.norm_psf(self.fake_psf)
        # input of bs x c x h x w => c x bs x hpad x wpad
        pad_w = self.opt.psf_size//2
        orig = F.pad(self.orig, ( pad_w, pad_w, pad_w, pad_w), mode='reflect').transpose(0,1)
        self.fake_blur = F.conv2d(orig,fake_psf_norm,groups=self.opt.batch_size).transpose(0,1)
        # weight: outxinxkxk shoud be 8x1xkxk ,orig should be 1x8xhxw.
        self.loss_G_cycle = self.criterionGenerate(self.fake_blur,self.real_blur) * self.opt.lambda3

        self.loss_G = self.loss_G_gen + self.loss_G_id +self.loss_G_GAN #+ self.loss_G_cycle
        self.loss_G.backward()

    def optimize_parameters(self):

        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                            ('G_gen', self.loss_G_gen.data[0]),
                            ('G_ID' , self.loss_G_id.data[0]),
                            ('G_cyc', self.loss_G_cycle.data[0]),
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0])
                            ])

    def get_total_loss(self):
        return self.loss_G_GAN.data[0] + self.loss_G_gen.data[0] + self.loss_G_id.data[0] +\
               self.loss_G_cycle.data[0] + self.loss_D_real.data[0] + self.loss_D_fake.data[0]

    def get_current_visuals(self):
        real_blur = tensor2im_tanh(self.real_blur.data)
        fake_blur = tensor2im_tanh(self.fake_blur.data)
        fake_psf = tensor2im_tanh(self.fake_psf.data)
        real_psf = tensor2im_tanh(self.real_psf.data)
        return OrderedDict([('real_blur', real_blur), ('fake_blur', fake_blur),('fake_psf', fake_psf), ('real_psf', real_psf)])


    def save_network(self,epoch,step,label='train'):
        file_name = self.opt.get_ckpt_name()
        if label is 'train':
            file_name = 'e{}_st{}_'.format(epoch,step) + file_name
        else:
            file_name = 'best_'.format(epoch, step) + file_name

        save_path = os.path.join(self.opt.ckpt_dir, file_name)
        model_dict = OrderedDict([('G',self.netG.cpu()),
                                  ('D',self.netD.cpu()),
                                  ('opt',self.opt),
                                  ('epoch',epoch),
                                  ('step',step),
                                  ('time','I dont Know')
                                  ])
        torch.save(model_dict, save_path)
        if self.opt.cudaid is not 0 and torch.cuda.is_available():
            self.netG.cuda()
            self.netD.cuda()

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def set_identity_psf(self,id_psf):
        id_psf = id_psf.zero_()
        ks = id_psf.size(2)
        id_psf[:,:,ks//2,ks//2] = 1

    def norm_psf(self,psf):
        # div by its sum of dim2,3
        bs = psf.size(0)
        ch = psf.size(1)
        wh = psf.size(2)
        psf.contiguous()
        sumtensor = psf.view(-1,wh*wh).sum(1).view(bs,ch).unsqueeze(2).unsqueeze(3).expand(bs,ch,wh,wh)
        return psf/sumtensor


opt = Option()
bmvc_data = BMVC_blur_psf_orig_Folder(opt)
bmvc_loader = DataLoader(bmvc_data, batch_size=opt.batch_size)
dataset_size = bmvc_data.__len__()
print("------------- loading dataset ---------------")
print("img num: %d" %(dataset_size) )
print("---------------------------------------------")

model = blindGAN(opt)
visualizer = draw.Visualizer(model.name())

total_steps = 0
best_loss = None
for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(bmvc_loader):
        iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter = total_steps - dataset_size * (epoch - 1)

        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.interval_vis == 0:
            visualizer.display_current_results(model.get_current_visuals())

        if total_steps % opt.interval_log == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, errors)

        total_loss = model.get_total_loss()
        if best_loss is None:
            best_loss = total_loss
        elif best_loss>total_loss:
            best_loss = total_loss
            model.save_network(epoch,total_steps,'best')

        if total_steps % opt.interval_save == 0:
            model.save_network(epoch, total_steps,'train')


    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()