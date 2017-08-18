import torch.nn as nn

class Option:
    def __init__(self):

        # option for G
        self.G_block_num = 9 # 8 by default
        self.G_bottle_scale =2 # 2(twice) by default
        self.G_norm_mode = 'IN' # 'IN' instance norm by default, 'BN' batch norm

        # option for D
        self.D_layer_nun = 2 # 3 by default
        self.D_usesigmoid = True
        self.D_norm_mode = 'BN' # 'BN' by default
        self.pool_size = 30

        self.psf_size = 29
        self.img_by_psf = 4 #=G_bottle_scale**2
        self.img_size = self.psf_size * self.img_by_psf
        self.color_mode = 'L' # 'L'by default,'RGB'

        # option for loss
        self.ganloss = 'mse' # mse(use lsgan), bce
        self.lambda1 = 1
        self.generate_loss = 'mse' # mse, L1
        self.lambda2 = 1
        self.identity_loss = 'l1' # mse,l1
        self.lambda3 = 1 #for cyc_loss

        # option for training
        self.train_dir = '/home/wcd/LinkToMyLib/Datas/BMVC_large_patches'
        self.dropout = 0.5
        self.optimize_mode = 'admm'
        self.lr = 1e-2
        self.niter = 100 # num of iter at starting learning rate')
        self.niter_decay = 100 # num of iter to linearly decay learning rate to zero')
        self.epoch = 2
        self.interval_eval = 200
        self.interval_log = 100
        self.interval_vis = 200
        self.batch_size = 8
        self.ckpt_dir = 'ckpt/'
        self.beta1 = 0.5 #momentum term of adam

        # option for testing
        self.test_dir = None

        # visiual setting
        self.vis_imgpath = '/home/wcd/LinkToMyLib/Datas/BMVC_large_patches/0007521_orig.png'

        self.cudaid = 1
        self.training_fin = False
        self.training_con = True

    def get_ckpt_name(self):
        name = 'blindGAN_Lgan_{self.ganloss}_Lid_{self.identity_loss}_Lgen{self.generate_loss}_' \
               'G{self.G_norm_mode}block{self.G_block_num}_D{self.D_norm_mode}layer{self.D_layer_nun}.ckpt'.format(self)
        #eg blindGAN_Lgan_mse_Lid_mse_Lgen_mse_GIN8_DBN3.ckpt
        return name

    def finish(self):
        self.training_fin = True
        self.training_con = False
