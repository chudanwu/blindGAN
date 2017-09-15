import torch
import torch.nn.init
import numpy as np

class TransformerNet(torch.nn.Module):
    def __init__(self, mode='RGB',resblock_num=8,skiplayer = 1,IN_flag = True,deconv_mode='PS'):
        super(TransformerNet, self).__init__()
        if mode is 'Y' or mode is 'L':
            channel = 1
        else :
            channel = 3
        # Initial convolution layers

        reschannel = 64
        # Residual layers
        #scale means times of double img size
        if IN_flag is True:
            self.resblocks = torch.nn.ModuleList([ResidualBlock_IN(reschannel) for resblock in range(resblock_num)])
            self.encoder = EncoderBlock_IN(channel,reschannel,scale=2)
            self.decoder = DecoderBlock_IN(reschannel,channel,scale=2,deconv_mode=deconv_mode)
        else:
            self.resblocks = torch.nn.ModuleList([ResidualBlock(reschannel) for resblock in range(resblock_num)])
            self.encoder = EncoderBlock(channel,reschannel,scale=2)
            self.decoder = DecoderBlock(reschannel,channel, scale=2,deconv_mode=deconv_mode)

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

class condition_TransformerNet(torch.nn.Module):
    def __init__(self, mode='RGB',resblock_num=8,skiplayer = 1,IN_flag = True,deconv_mode='PS'):
        super(condition_TransformerNet, self).__init__()
        if mode is 'Y' or mode is 'L':
            channel = 1+5
        else :
            channel = 3+5
        # Initial convolution layers

        reschannel = 64
        # Residual layers
        #scale means times of double img size
        if IN_flag is True:
            self.resblocks = torch.nn.ModuleList([ResidualBlock_IN(reschannel) for resblock in range(resblock_num)])
            self.encoder = EncoderBlock_IN(channel,reschannel,scale=2)
            self.decoder = DecoderBlock_IN(reschannel,channel-5,scale=2,deconv_mode=deconv_mode)
        else:
            self.resblocks = torch.nn.ModuleList([ResidualBlock(reschannel) for resblock in range(resblock_num)])
            self.encoder = EncoderBlock(channel,reschannel,scale=2)
            self.decoder = DecoderBlock(reschannel,channel-5, scale=2,deconv_mode=deconv_mode)

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




    def forward(self, X,c):
        #print([X.size(),c.size()])
        resin = self.encoder(torch.cat((X,c),1))
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


def create_condition(batch_n,batch_params,h,w): # batch,params
    if batch_n is not batch_params.shape[0]:
        print('error!')
    condition_class = torch.zeros((batch_params.shape[0],2,h,w))
    condition_param = torch.zeros((batch_params.shape[0],3, h, w))
    for i in range(batch_params.shape[0]): # in each batch
        params = batch_params[i]
        gauss = params[0]
        motion_len = params[1]
        motion_ang = params[2]
        if gauss is not None and gauss > 0:
            if motion_len is not None:  # gauss + motion
                blurclass = np.array([[0.5], [0.5]])
            else:  # gauss
                blurclass = np.array([[1], [0]])
                motion_len = 0
                motion_ang = 0
        else:
            if motion_len is not None and motion_len > 0:  # motion
                blurclass = np.array([[0], [1]])
                gauss = 0
            else:  # none
                blurclass = np.array([[0], [0]])
                gauss = 0
                motion_len = 0
                motion_ang = 0
        condition_class[i] = torch.from_numpy(blurclass).unsqueeze(1).expand(2,h,w).contiguous()
        condition_param[i] = torch.from_numpy(np.array([gauss,motion_len,motion_ang])).unsqueeze(1).unsqueeze(1).expand(3,h, w).contiguous()
    return torch.cat((condition_class,condition_param),1).contiguous()

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)


    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock_IN(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock_IN, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out = out + residual
        return out


class ResidualBlock(torch.nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        return out


class EncoderBlock_IN(torch.nn.Module):
    def __init__(self, in_c,out_c,scale=2):
        super(EncoderBlock_IN, self).__init__()
        self.reschannel = out_c
        self.convlist = torch.nn.ModuleList()
        self.inlist = torch.nn.ModuleList()
        self.relu = torch.nn.ReLU()
        self.convlist.append(ConvLayer(in_c, self.reschannel//(2**scale), kernel_size=9, stride=1))
        self.inlist.append(torch.nn.InstanceNorm2d(self.reschannel//(2**scale), affine=True))
        for i in range(scale,0,-1):
            c_in = self.reschannel//(2**i)
            self.convlist.append(ConvLayer(c_in, c_in*2, kernel_size=3, stride=2))
            self.inlist.append(torch.nn.InstanceNorm2d(c_in*2, affine=True))

    def forward(self, x):
        out = x
        for i,conv in enumerate(self.convlist):
            out = conv(out)
            out = self.inlist[i](out)
            out = self.relu(out)
        return out


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_c,out_c,scale=2):
        super(EncoderBlock, self).__init__()
        self.reschannel = out_c
        self.convlist = torch.nn.ModuleList()
        self.relu = torch.nn.ReLU()
        self.convlist.append(ConvLayer(in_c, self.reschannel//(2**scale), kernel_size=9, stride=1))
        for i in range(scale,0,-1):
            c_in = self.reschannel//(2**i)
            self.convlist.append(ConvLayer(c_in, c_in*2, kernel_size=3, stride=2))

    def forward(self, x):
        out = x
        for i,conv in enumerate(self.convlist):
            out = conv(out)
            out = self.relu(out)
        return out



class DecoderBlock_IN(torch.nn.Module):
    def __init__(self, in_c,out_c ,scale=2,deconv_mode='PS'):
        super(DecoderBlock_IN, self).__init__()
        self.convlist = torch.nn.ModuleList()
        self.inlist = torch.nn.ModuleList()
        self.relu = torch.nn.ReLU()
        c_in = in_c
        for i in range(scale):
            if deconv_mode is 'PS':
                c_out = c_in//(scale**2)
            else:
                c_out = c_in // 2
            self.convlist.append(ConvPSLayer(c_in, c_out, kernel_size=3, stride=1,upsample=2))
            self.inlist.append(torch.nn.InstanceNorm2d(c_out, affine=True))
            c_in = c_out
        self.lastconv = ConvLayer(c_out, out_c, kernel_size=3, stride=1)

    def forward(self, x):
        out = x
        for i,conv in enumerate(self.convlist):
            out = conv(out)
            out = self.inlist[i](out)
            out = self.relu(out)
        out = self.lastconv(out)
        return out


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_c,out_c,scale=2,deconv_mode='PS'):
        super(DecoderBlock, self).__init__()
        c_in = in_c
        self.convlist = torch.nn.ModuleList()
        self.relu = torch.nn.ReLU()
        for i in range(scale):
            if deconv_mode is 'PS':
                c_out = c_in//(scale**2)
            else:
                c_out = c_in // 2
            self.convlist.append(ConvPSLayer(c_in, c_out, kernel_size=3, stride=1,upsample=2))
            c_in = c_out
        self.lastconv = ConvLayer(c_in, out_c, kernel_size=3, stride=1)

    def forward(self, x):
        out = x
        for i,conv in enumerate(self.convlist):
            out = conv(out)
            out = self.relu(out)
        out = self.lastconv(out)
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
