
import time
import os
import torch.nn.functional as F
import torch
import  torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from utils import print_params_num, ImagePool, tensor2im, tensor2im_tanh
import draw
from dataFolder import vaeSRFolder
import models
from options import vaeSROption

loss_dict = {'mse':nn.MSELoss, 'l1':nn.L1Loss, 'bce':nn.BCELoss}
channel_of_colormode = {'L':1,'Y':1,'RGB':3}

# blindVAE trainer
class blindVAE():
    def name(self):
        return 'blindVAE'

    def __init__(self, opt,isTrain=True,isSeperate=True,fromPretrain=False):

        self.isTrain = isTrain
        self.isSeperate = isSeperate
        self.fromPretrain = fromPretrain
        self.opt = opt
        if self.opt.cudaid is not None:
            torch.cuda.set_device(self.opt.cudaid)

        # define tensor type in/not in cuda
        self.Tensor = torch.cuda.FloatTensor if opt.cudaid else torch.FloatTensor
        self.tensorLR = self.Tensor(opt.batch_size,channel_of_colormode[opt.color_mode],
                                   opt.img_size,opt.img_size)
        self.tensorHR = self.tensorLR.clone()
        self.tensorP = self.Tensor(opt.batch_size, opt.param_num,opt.img_size, opt.img_size)

        # load/define networks
        self.netSR = models.condition_SRNet(opt.param_num,mode=opt.color_mode,resblock_num=opt.SR_block_num,
                                           norm_mode=opt.SR_norm_mode,drop_out=opt.dropout,
                                           deconv_mode='US',bottle_scale=opt.SR_bottle_scale)
        self.netP = models.condition_LRNet(opt.param_num,mode=opt.color_mode,resblock_num=opt.P_block_num,
                                            norm_mode = opt.P_norm_mode,drop_out=opt.dropout)
        # define loss functions
        self.criterionSR = loss_dict[opt.SR_loss]()
        self.criterionP = loss_dict[opt.P_loss]()

        if self.opt.cudaid is not 0:
            self.netSR = self.netSR.cuda()
            self.netP = self.netP.cuda()


        if self.isTrain:
            if self.fromPretrain:  # or opt.continue_train:
                self.load_network(opt.seperate_model)
            self.old_lr = opt.lr

            self.netSR.train()
            self.netP.train()
            # initialize optimizers
            self.optimizer_SR = torch.optim.Adam(self.netSR.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_P = torch.optim.Adam(self.netP.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.optimize_mode is not 'admm':
                raise ValueError('optimize_mode: %s undifine'%(opt.optimize_mode))
        else:
            self.netP.eval()
            self.netSR.eval()

        print('------------ Networks initialized -------------')
        print_params_num(self.netSR)
        print_params_num(self.netP)
        print('-----------------------------------------------')

    def set_input(self, input): #HR, LR, params
        input_hr = input[0]
        input_lr = input[1]
        input_p = input[2]
        self.tensorLR.resize_(input_lr.size()).copy_(input_lr)
        self.tensorHR.resize_(input_hr.size()).copy_(input_hr)
        input_p = self.create_condition(input_p)
        self.tensorP.resize_(input_p.size()).copy_(input_p)

    def create_condition(self, batch_params):  # n,[g,m_x,m_y] unnormalize=> n,3,h,w normalize and pixelwise
        if self.opt.batch_size is not batch_params.shape[0]:
            print('error!:batch size is not consistance with patch-params')
        condition_param = torch.zeros((self.opt.batch_size, self.opt.param_num, self.tensorLR.shape[2], self.tensorLR.shape[3]))
        for i in range(batch_params.shape[0]):  # in each batch
            params = batch_params[i]
            gauss = params[0]
            motion_x = params[1]
            motion_y = params[2]
            if gauss is None or gauss is 0:
                gauss = 0
            if motion_x is 0 and motion_y is 0:
                motion_x = 0
                motion_y = 0
            gauss = gauss / self.opt.gauss_max
            motion_x = motion_x / self.opt.motion_x_max
            motion_y = motion_y / self.opt.motion_y_max
            condition_param[i] = torch.from_numpy(np.array([gauss, motion_x, motion_y])).unsqueeze(1).unsqueeze(
                1).expand(self.opt.param_num, self.tensorLR.shape[2], self.tensorLR.shape[3]).contiguous()
        # print(condition_param.shape)
        return condition_param.contiguous()

    def forward(self):

        self.LR = Variable(self.tensorLR)
        self.fake_p = self.netP(self.LR)
        self.real_p = Variable(self.tensorP)
        if self.isSeperate:
            self.SR = self.netSR(self.LR.detach(), self.real_p) # seperate training
        else:
            self.SR = self.netSR(self.LR.detach(),self.fake_p.detach()) # combine training

    # no backprop gradients
    def validate(self):

        self.LR = Variable(self.tensorLR, volatile=True)
        HR = Variable(self.tensorHR,volatile=True)
        self.real_p = Variable(self.tensorP,volatile=True)
        self.fake_p = self.netP(self.LR)
        self.SR = self.netSR(self.LR,self.fake_p)
        self.loss_sr = self.criterionSR(self.SR, HR)
        self.loss_p = self.criterionP(self.fake_p, self.real_p)

    def backward(self): #

        HR = Variable(self.tensorHR)
        if self.opt.tanhoutput:
            self.SR.clamp(-1,1)
        else:
            self.SR.clamp(0, 1)
        self.loss_sr = self.criterionSR(self.SR,HR) # this sr is from fake_p
        self.loss_sr.backward() # should not affect netP

        self.loss_p = self.criterionP(self.fake_p, self.real_p)
        self.loss_p.backward()

    def optimize_parameters(self):

        self.forward()
        self.optimizer_SR.zero_grad()
        self.optimizer_P.zero_grad()
        self.backward()
        self.optimizer_SR.step()
        self.optimizer_P.step()


    def get_current_errors(self):
        return OrderedDict([('ENCODER', self.loss_p.data[0]),
                            ('DECODER', self.loss_sr.data[0])
                            ])

    def get_total_loss(self):
        return  self.loss_p.data[0] + self.loss_sr.data[0]

    def get_current_visuals(self):
        if self.opt.tanhoutput:
            lr = tensor2im_tanh(self.tensorLR)
            hr = tensor2im_tanh(self.tensorHR)
            sr = tensor2im_tanh(self.SR.data)
        else:
            lr = tensor2im(self.tensorLR)
            hr = tensor2im(self.tensorHR)
            sr = tensor2im(self.SR.data)
        return OrderedDict([('lr', lr), ('hr', hr),('sr', sr)])


    def save_network(self,epoch,step,label='train'):
        file_name = self.opt.get_ckpt_name()
        if self.isSeperate:
            combine_or_seperate = 'sep'
        else:

            combine_or_seperate = 'com'
        if label is 'train':
            file_name = combine_or_seperate + 'e{}_st{}_'.format(epoch,step) + file_name
        else:
            file_name = combine_or_seperate + 'best_'.format(epoch, step) + file_name

        save_path = os.path.join(self.opt.ckpt_dir, file_name)
        model_dict = OrderedDict([('P',self.netP.cpu()),
                                  ('SR',self.netSR.cpu()),
                                  ('opt',self.opt),
                                  ('epoch',epoch),
                                  ('step',step),
                                  ('time','I dont Know')
                                  ])
        torch.save(model_dict, save_path)
        if self.opt.cudaid is not 0 and torch.cuda.is_available():
            self.netSR.cuda()
            self.netP.cuda()

    def load_network(self, filename):
        model_dict = torch.load(filename)
        self.netP = model_dict['P']
        self.netSR = model_dict['SR']
        if self.isTrain:
            self.netP.train()
            self.netSR.train()
        else:
            self.netP.eval()
            self.netSR.eval()
        if self.opt.cudaid is not 0:
            self.netP = self.netP.cuda()
            self.netSR = self.netSR.cuda()

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_SR.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_P.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

def train(isTrain=True,isSeperate=True,fromPretrain=False):
    opt = vaeSROption()
    init_lr = opt.lr
    randomblur_data = vaeSRFolder(opt)
    randomblur_loader = DataLoader(randomblur_data, batch_size=opt.batch_size)
    dataset_size = randomblur_data.__len__()
    print("------------- loading dataset ---------------")
    print("img num: %d" %(dataset_size) )
    print("---------------------------------------------")

    model = blindVAE(opt,isTrain=isTrain,isSeperate=isSeperate,fromPretrain=fromPretrain)
    visualizer = draw.Visualizer(model.name())

    total_steps = 0
    best_loss = None

    # seperate training
    if not fromPretrain:
        print("-------------Seperate training---------------")
        for epoch in range(1, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()
            for i, data in enumerate(randomblur_loader):
                if len(data[0]) is not opt.batch_size:
                    continue
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


    print('--------------Combine training---------------')
    # conbime training
    model.old_lr = init_lr
    model.isSeperate = False
    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        for i, data in enumerate(randomblur_loader):
            if len(data[0]) is not opt.batch_size:
                continue
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

if __name__ == "__main__":
    train(isTrain=True,isSeperate=False,fromPretrain=True)