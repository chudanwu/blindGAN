import torch
import torch.nn.init
from torch.autograd import Variable

class GNet(torch.nn.Module):
    '''
    GNet for generate similar context and same size(channel,height,width)
    '''
    def __init__(self, mode='RGB',resblock_num=8,skiplayer = 1,norm_mode = 'IN',deconv_mode='PS',bottle_scale=2,dropout=None):
        super(GNet, self).__init__()
        if mode is 'Y' or mode is 'L':
            channel = 1
        else :
            channel = 3

        # Residual layers
        reschannel = 64
        self.resblocks = torch.nn.ModuleList([ResidualBlock(reschannel,norm_mode=norm_mode,drop_out=dropout) for resblock in range(resblock_num)])

        self.encoder = EncoderBlock(channel,reschannel,downscale=bottle_scale,norm_mode=norm_mode)
        self.decoder = DecoderBlock(reschannel,channel,upscale=bottle_scale,norm_mode=norm_mode,deconv_mode=deconv_mode)

        self.skiplayer = skiplayer
        if skiplayer is not 0:
            self.mergeconv = ConvLayer(2 * reschannel, reschannel, 3, 1)

        '''暂时只skip一次，用不上下面
        if self.skiplayer > 0:
            self.mergeconv = torch.nn.ModuleList()
            for latter in range(resblock_num // 2, resblock_num, 1):
                former = resblock_num - 1 - latter
                if former in self.skiplayer :
                    print("skiplayer:"+str(former) + ":" + str(latter))
                    self.mergeconv.append( ConvLayer(2*reschannel,reschannel,3,1) )
                    '''
    def forward(self, X):
        resin = self.encoder(X)
        y = resin
        for i, res in enumerate(self.resblocks):
            y = res(y)

        if self.skiplayer is not 0:
            y = torch.cat((resin,y),1)
            y = self.mergeconv(y)

        y = torch.cat((resin, y), 1)
        y = self.mergeconv(y)
        y = self.decoder(y)
        y = y + X
        return y

class GpsfNet():
    '''
        GNet for generate unnorm psf of certain size(1x29x29), value in (0,1)
        '''

    def __init__(self, mode='RGB', out_c=1,resblock_num=8, downscale=2,norm_mode='IN',drop_out=None):
        super(GpsfNet, self).__init__()
        if mode is 'Y' or mode is 'L':
            in_c = 1
        else:
            in_c = 3
        # Initial convolution layers

        # Residual layers
        reschannel = 64
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(reschannel, norm_mode=norm_mode,drop_out=drop_out) for resblock in range(resblock_num)])

        self.encoder = EncoderBlock(in_c, reschannel, downscale=downscale, norm_mode=norm_mode)
        self.activelayer =torch.nn.Sequential(
            [
                ConvLayer(reschannel, out_c, kernel_size=3, stride=1),
                torch.nn.Sigmoid()# when output is norm to(mean=0.5,mean=0.5)->TANH
            ])

    def forward(self, X):
        resin = self.encoder(X)
        y = resin
        for i, res in enumerate(self.resblocks):
            y = res(y)
        y = self.activelayer(y)
        return y

class DNet(torch.nn.Module):
    '''
    DNet for certain input size
    '''
    def __init__(self, in_c, insize=29, out_c=1, n_layers=3, norm_mode = 'IN',use_sigmoid=True):
        # in_c should be channel of input, either G()_c or G()_c + Condiction_c
        super(DNet, self).__init__()
        self.insize = insize
        if (norm_mode is not 'IN' or norm_mode is not 'BN'):
            raise ValueError('norm_mode value [%s] is undifined' %norm_mode)
        feat_c = 64
        sequence = [
            ConvLayer(in_c, feat_c, kernel_size=4, stride=2),
            torch.nn.LeakyReLU(0.2, True)
        ]

        feat_c_mult = 1
        feat_c_mult_prev = 1
        for n in range(1, n_layers):
            feat_c_mult_prev = feat_c_mult
            feat_c_mult = min(2**n, 8)
            sequence += [
                ConvLayer(feat_c*feat_c_mult_prev, feat_c * feat_c_mult,
                          kernel_size=4, stride=2),
                create_norm_layer(feat_c * feat_c_mult,norm_mode),
                torch.nn.LeakyReLU(0.2, True)
            ]

        # 2nd last conv layer stride=1
        feat_c_mult_prev = feat_c_mult
        feat_c_mult = min(2**n_layers, 8)
        sequence += [
            ConvLayer(feat_c * feat_c_mult_prev, feat_c * feat_c_mult,kernel_size=4, stride=1),
            create_norm_layer(feat_c * feat_c_mult,norm_mode),
            torch.nn.LeakyReLU(0.2, True)
        ]

        # last conv layer out_c=1 and maybe active by sigmoid to output prob
        sequence += [ConvLayer(feat_c * feat_c_mult, out_c, kernel_size=4, stride=1)]
        if use_sigmoid:
            # last layer end with sigmoid
            sequence += [torch.nn.Sigmoid()]

        self.model = torch.nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input

class GANLoss(torch.nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        # variable与value不同，因为Variable有shape，当Dnet输出是Patch时，要把target的shape弄到和Dnet的输出一样
        # 所以若用PatchGAN，则target Var需要create,
        # 但是如果是L2GAN由于mse算出来是一个值(mse范围是0-max^2,这里max是mse输入元素的最大值)，那么target var也没有size，又不用creat了
        # variable is different from value, target value is a float num without size,
        # while target variable has same shape as the output of Dnet,
        # so to use patchGAN, creating the target var is necessary.
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = torch.nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            # input of Dnet is from dataset
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                # which means input!=target value,
                # or their size is inconsistent
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)# target doesnt need grad
            target_tensor = self.real_label_var
        else:
            # input od Dnet is from Gnet output pool
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        # return loss according to the chosen target
        target_tensor = self.get_target_tensor(input, target_is_real)
        # target_tensor is instance of 'Variable'
        return self.loss(input, target_tensor)



class ConvLayer(torch.nn.Module):
    '''convolution with reflection padding'''
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
        introduced in: https://arxiv.org/abs/1512.03385
        recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """
    def __init__(self, channels, norm_mode=None, drop_out=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = torch.nn.ReLU()
        self.norm_flag = False
        self.dropout_flag =False
        if norm_mode is not None:
            self.norm1 = create_norm_layer(channels, norm_mode=norm_mode)
            self.norm2 = create_norm_layer(channels, norm_mode=norm_mode)
            self.norm_flag = True
        if drop_out is not None:
            self.dropout = torch.nn.Dropout2d(drop_out)
            self.dropout_flag = True

    def forward(self, x):

        out = self.conv1(x)
        if self.norm_flag:
            out = self.norm1(out)
        out = self.relu(out)
        if self.dropout_flag:
            out = self.dropout(out)
        out = self.conv2(out)
        if self.norm_flag:
            out = self.norm2(out)
        out = self.relu(out)

        out = out + x
        return out


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, norm_mode='IN',downscale=2):
        # downscale means: half(determine by 'stride=2') the width and height downscale times
        super(EncoderBlock, self).__init__()
        self.convlist = torch.nn.ModuleList()
        self.convlist.append(ConvLayer(in_c, out_c//(2**downscale), kernel_size=9, stride=1))
        self.normlist = None if norm_mode is None else torch.nn.ModuleList()
        if norm_mode is not None:
            self.normlist.append(create_norm_layer(out_c//(2**downscale),norm_mode=norm_mode))

        for i in range(downscale,0,-1):
            layer_in_c = out_c//(2**i)
            self.convlist.append(ConvLayer(layer_in_c, layer_in_c*2, kernel_size=3, stride=2))
            self.normlist.append(torch.nn.InstanceNorm2d(layer_in_c*2, affine=True))

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = x
        for i,conv in enumerate(self.convlist):
            out = conv(out)
            if self.normlist is not None:
                out = self.normlist[i](out)
            out = self.relu(out)
        return out


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_c,out_c,norm_mode='IN',upscale=2,deconv_mode='PS',upsample = 2):
        # upscale means: double(determine by 'upsample=2') the width and height upscale times
        super(DecoderBlock, self).__init__()
        self.convlist = torch.nn.ModuleList()
        self.normlist = None if norm_mode is None else torch.nn.ModuleList()
        self.relu = torch.nn.ReLU()

        layer_in_c = in_c
        for i in range(upscale):
            if deconv_mode is 'PS':
                layer_out_c = layer_in_c//(upsample**2)
                self.convlist.append(ConvPSLayer(layer_in_c, layer_out_c, kernel_size=3, stride=1, upsample=upsample))
            else:
                layer_out_c = layer_in_c // 2
                self.convlist.append(UpsampleConvLayer(layer_in_c, layer_out_c, kernel_size=3, stride=1,upsample=upsample))
                if norm_mode is not None:
                    self.normlist.append(create_norm_layer(layer_out_c,norm_mode))
            layer_in_c = layer_out_c
        self.lastlayer=torch.nn.Sequential(
            [
                ConvLayer(layer_out_c, out_c, kernel_size=3, stride=1),
                torch.nn.ReLU()
            ])

    def forward(self, x):
        out = x
        for i,conv in enumerate(self.convlist):
            out = conv(out)
            if self.normlist is not None:
                out = self.normlist[i](out)
            out = self.relu(out)
        self.lastlayer(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.UpsamplingNearest2d(scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class ConvPSLayer(torch.nn.Module):
    """
    conv of (in,outxuxu,k,s)
    follow by
    pixel shuffle of scalling factor (u)
    output is (bs out hxu wxu)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(ConvPSLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels*(upsample**2), kernel_size, stride)
        if upsample:
            self.pixelshuffle_layer = torch.nn.PixelShuffle(upsample)

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.upsample:
            out = self.pixelshuffle_layer(out)
        return out


def create_norm_layer(channel,norm_mode='IN'):
    # ##affine setting is deferent!
    if norm_mode == 'IN':
        norm_layer = torch.nn.InstanceNorm2d(channel, affine=False)
    elif norm_mode == 'BN':
        norm_layer = torch.nn.BatchNorm2d(channel, affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_mode)
    return norm_layer


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram