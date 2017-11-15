import torch.nn as nn

class Option:
    def __init__(self):
        # option for G
        self.G_block_num = 9  # 8 by default
        self.G_bottle_scale = 2  # 2(twice) by default
        self.G_norm_mode = 'IN'  # 'IN' instance norm by default, 'BN' batch norm

        # option for D
        self.D_layer_nun = 2  # 3 by default
        self.D_usesigmoid = True
        self.D_norm_mode = 'BN'  # 'BN' by default
        self.pool_size = 30

        self.psf_size = 29
        self.img_by_psf = 4
        self.img_size = self.psf_size * self.img_by_psf
        self.color_mode = 'L'  # 'L'by default,'RGB'


        # option for loss
        self.ganloss = 'mse'  # mse(use lsgan), bce
        self.lambda1 = 10
        self.generate_loss = 'l1'  # mse, L1
        self.lambda2 = 1
        self.identity_loss = 'l1'  # mse,l1
        self.lambda3 = 1  # for cyc_loss

        # option for training
        self.train_dir = '/home/wcd/LinkToMyLib/Datas/BMVC_large_patches'
        self.dropout = None
        self.optimize_mode = 'admm'
        self.lr = 1e-3
        self.niter = 100  # num of iter at starting learning rate')
        self.niter_decay = 100  # num of iter to linearly decay learning rate to zero')
        self.epoch = 2
        self.interval_eval = 200
        self.interval_log = 100
        self.interval_vis = 200
        self.interval_save = 1000
        self.batch_size = 4
        self.ckpt_dir = 'ckpt/'
        self.beta1 = 0.5  # momentum term of adam

        # option for testing
        self.test_dir = None

        # visiual setting
        self.vis_imgpath = '/home/wcd/LinkToMyLib/Datas/BMVC_large_patches/0007521_orig.png'

        self.cudaid = 1
        self.training_fin = False
        self.training_con = True



    def get_ckpt_name(self):
        name = 'blindGAN_Lgan_{}_Lid_{}_Lgen{}_G{}block{}_D{}layer{}.ckpt'.format(
            self.ganloss,self.identity_loss,self.generate_loss,self.G_norm_mode,self.G_block_num,self.D_norm_mode,self.D_layer_nun)
        #eg blindGAN_Lgan_mse_Lid_mse_Lgen_mse_GIN8_DBN3.ckpt
        return name

    def finish(self):
        self.training_fin = True
        self.training_con = False

class vaeSROption:
    def __init__(self):
        # option for SR
        self.SR_block_num = 16  # 8 by default
        self.SR_bottle_scale = 2  # 2(twice) by default
        self.SR_norm_mode = 'IN'  # 'IN' instance norm by default, 'BN' batch norm

        # option for P
        self.P_block_num = 8  # 8 by default
        self.P_norm_mode = 'BN'

        self.img_size = 128
        color_mode_dict = {'L':1,'RGB':3,'YCbCr':3,'LG':2}  # 'L'by default,'RGB'
        self.color_mode='L'
        self.channelofimg = color_mode_dict[self.color_mode]
        self.params = ['g_sig','m_x','m_y','d_rad']
        self.param_num = len(self.params)  # gauss_sig,gauss_len,motion_angle,motion_len
        self.gauss_len_max = 3 #[3,5,7] startfrom 3 max is 2max+1
        self.gauss_sig_max = 3.0
        self.gauss_max = 3.0
        self.motion_len_max = 10
        self.motion_angle_max = 180
        self.motion_x_max = self.motion_len_max/2
        self.motion_y_max = self.motion_len_max/2
        self.defocus_rad_max = 5

        self.block_scale = 3
        self.tanhoutput = True #tanh/conv2d:-1~1(use norm) sigmoid:0-1
        # input is -1~1,then output is -1~1,both 'tanhoutput'or'conv without activation' can be chosen

        # option for loss
        self.SR_loss = 'mse'  # mse, l1
        self.P_loss = 'mse'  # mse, l1

        # option for training
        self.train_dir = '/home/wcd/LinkToMyLib/Datas/train2014'
        self.save_train_img = False  # true
        self.save_train_img_dir = '/media/library/wcd/Datas/coco2blur'
        self.val_dir = "/media/library/wcd/Datas/BSDS/BSDS300/images/train"
        self.dropout = None
        self.optimize_mode = 'admm'
        self.lr = 1e-3
        self.niter = 5  # num of iter at starting learning rate')
        self.niter_decay = 5  # num of iter to linearly decay learning rate to zero')
        self.interval_eval = 200
        self.interval_log = 100
        self.interval_vis = 200
        self.interval_save = 80000
        self.batch_size = 8
        self.ckpt_dir = '/media/library/wcd/Models/Train_deblur/patch_in/condition_p'
        self.beta1 = 0.5  # momentum term of adam

        # option for testing
        self.test_dir = None

        # visiual setting
        self.vis_imgpath = '/home/wcd/LinkToMyLib/Datas/BMVC_large_patches/0007521_orig.png'

        self.pretrain_model = '/home/wcd/Desktop/to_neural_style/ckpt/come2_st160000_block_vaeSR_gsig_3.0_glen_3_mang_89_mlen8_SRIN_block4_PBN_block8.model'
        self.cudaid = 1
        self.training_fin = False
        self.training_con = True

    def get_ckpt_name(self):
        name = 'vaeSR_gsig_{}_glen_{}_mang_{}_mlen{}_SR{}_block{}_P{}_block{}.model'.format(
            self.gauss_sig_max,self.gauss_len_max,self.motion_angle_max,self.motion_len_max,self.SR_norm_mode,self.SR_block_num,self.P_norm_mode,self.P_block_num)
        return name

    def finish(self):
        self.training_fin = True
        self.training_con = False