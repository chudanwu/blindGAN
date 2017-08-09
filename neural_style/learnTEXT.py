import argparse
import os
import sys
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import utils
import draw
from LRnet import TransformerNet
import visdom


# learn LR ,but reference is itself
def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x_HR: x_HR.mul(255))
    ])

    # load training folder
    train_dataset = utils.BMVCFolder(args.dataset, args.HR_size, LR_scale=None,
                                         transform=transform, target_transform=transform, mode=args.mode)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    '''
    # load validate folder
    val_dataset = utils.BMVCFolder(args.valset, args.HR_size, LR_scale=None,
                                       transform=transform, target_transform=transform, mode=args.mode)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    '''

    transformer = TransformerNet(args.mode, resblock_num=args.resblobk_num, skiplayer=0,IN_flag=False, deconv_mode=args.deconv_mode)
    optimizer = Adam(transformer.parameters(), args.lr)


    # define loss function:
    if args.lossfun is "mse":
        # 数量级：1e2
        lossfun = torch.nn.MSELoss()
    elif args.lossfun is "l1":
        #数量级：1e1
        lossfun = torch.nn.L1Loss()
    else:
        lossfun = torch.nn.L1Loss()

    # 数量级：1e2~1e3
    tvloss = utils.TVLoss()
    l1reg = utils.L1Reg()
    mse = torch.nn.MSELoss()


    # 读取HR用来测试和看
    if args.mode is 'Y' :
        ref_LR_y, ref_LR_b, ref_LR_r = utils.load_HR_image(args.style_image, mode='YCbCr').split()
        ref_LR = ref_LR_y
    else:
        ref_LR = utils.load_HR_image(args.style_image, mode=args.mode)
        ref_LR_b = None
        ref_LR_r = None

    # 用来测试的HR
    ref_LR = transform(ref_LR)
    if args.cuda:
        ref_LR = ref_LR.cuda()
        transformer.cuda()
    ref_LR = Variable(ref_LR.unsqueeze(0), requires_grad=False)
    if args.normalizebatch_flag is True:
        ref_LR = utils.normalize_batch(ref_LR)

    # 用来看的HR
    if args.mode is not "L":
        ref_HR_RGB = transform(utils.load_HR_image('/home/wcd/LinkToMyLib/Datas/BMVC_large_patches/0022832_orig.png', mode='RGB'))
        ref_LR_RGB = transform(utils.load_HR_image(args.style_image, mode='RGB'))
    else:
        ref_HR_RGB = transform(utils.load_HR_image('/home/wcd/LinkToMyLib/Datas/BMVC_large_patches/0022832_orig.png', mode='L'))
        ref_LR_RGB = transform(utils.load_HR_image(args.style_image, mode='L'))
    viz = visdom.Visdom(env='wcd')
    HR_win = viz.image(
        ref_HR_RGB.squeeze(0),
        opts=dict(title='HR')
    )
    LR_win = viz.image(
        ref_LR_RGB.squeeze(0),
        opts=dict(title='LR', caption="LR: mode{},len:{},ang:{},g{}".format(
                        args.mode,args.motion_len,args.motion_angel,args.gauss))
    )
    gen_win = viz.image(
        ref_LR_RGB.squeeze(0),
        opts=dict(title='genHR')
    )

    for e in range(args.epochs):

        transformer.train()
        agg_content_loss = 0.
        count = 0
        loss_win = None
        bestloss = None

        for batch_id, (x_HR, x_LR) in enumerate(train_loader):
            n_batch = len(x_HR)
            count += n_batch
            optimizer.zero_grad()

            if args.cuda:
                x_HR, x_LR = x_HR.cuda(), x_LR.cuda()
            x_HR, x_LR = Variable(x_HR), Variable(x_LR)
            if args.normalizebatch_flag is True:
                #x_HR = utils.normalize_batch(x_HR)
                x_LR = utils.normalize_batch(x_LR)

            x_gen = transformer(x_LR).clamp(0, 255)

            if args.normalizebatch_flag is True:
                x_gen = utils.unnormalize_batch(x_gen).clamp(0, 255)

            # loss:
            content_loss = args.content_weight * lossfun(x_gen, x_HR)
            tv_loss = torch.abs(tvloss(x_gen)-tvloss(x_HR))
            l1reg_loss = l1reg(x_gen)
            loss = content_loss + l1reg_loss

            loss.backward()
            optimizer.step()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = ("{}\tEpoach {}:\t[{}/{}]\t"+args.lossfun+"loss: {:.6f}\t tvloss: {:.6f}\t l1reg: {:.6f}").format(
                    time.ctime(), e + 1, count, len(train_dataset), content_loss.data[0] / n_batch , tv_loss.data[0], l1reg_loss.data[0]
                )
                print(mesg)

            if (batch_id + 1) % args.plot_interval == 0:

                loss = (content_loss.data[0] / n_batch, l1reg_loss.data[0])
                loss_win = draw.drawdifloss(viz,loss_win,batch_id,loss,[args.lossfun,'l1_reg'])


            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.log_interval == 0:
                transformer.eval()

                if args.cuda:
                    transformer.cpu()
                utils.save_model(transformer, args, "none", e, batch_id)
                if args.cuda:
                    transformer.cuda()
                transformer.train()

                img = transformer(ref_LR)
                if args.normalizebatch_flag is True:
                    img = utils.unnormalize_batch(img)
                img = draw.var2imgnp(img)
                ref_gen = draw.imgnp_convert(img, args.mode, Cb=ref_LR_b, Cr=ref_LR_r)
                viz.image(
                    ref_gen,
                    win=gen_win,
                    opts=dict(title='genHR', caption="genHR: mode{},len:{},ang:{},g{}".format(
                        args.mode,args.motion_len,args.motion_angel,args.gauss))
                )
                transformer.train()

    # save model
    transformer.eval()
    if args.cuda:
        transformer.cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    content_image = utils.load_HR_image(args.content_image, size=args.HR_size)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda X: X.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)
    if args.cuda:
        content_image = content_image.cuda()
    content_image = Variable(content_image, volatile=True)

    style_model = TransformerNet()
    style_model.load_state_dict(torch.load(args.model))
    if args.cuda:
        style_model.cuda()
    output = style_model(content_image)
    if args.cuda:
        output = output.cpu()
    output_data = output.data[0]
    utils.save_image(args.output_image, output_data)


def validate(args, net, val_loader, loss_function):
    net.eval()
    agg_mse_loss = 0
    count = 0
    for batch_id, (v_HR, v_LR) in enumerate(val_loader):
        n_batch = len(v_HR)
        count += n_batch
        if args.cuda:
            v_HR, v_LR = v_HR.cuda(), v_LR.cuda()
        v_HR, v_LR = Variable(v_HR, requires_grad=False), Variable(v_LR, requires_grad=False)
        if args.normalizebatch_flag is True:
            v_HR = utils.normalize_batch(v_HR)
            # v_LR = utils.normalize_batch(v_LR)
        v_gen = net(v_LR)
        if args.normalizebatch_flag is True:
            v_gen = utils.unnormalize_batch(v_gen)
        content_loss = args.content_weight * loss_function(v_gen, v_HR)
        agg_mse_loss += content_loss.data[0]
    net.train()
    return agg_mse_loss / count


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, default='/home/wcd/LinkToMyLib/Datas/train2014',
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/timg.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, default="ckpt",
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, default=1,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, default='images/content-images/amber.jpg',
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, default="output-images",
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, default=1,
                                 help="set it to 1 for running on GPU, 0 for CPU")

    args = main_arg_parser.parse_args()

    args.subcommand = "train"
    args.cuda = 1
    args.save_model_dir = "ckpt"
    args.dataset = '/home/wcd/LinkToMyLib/Datas/BMVC_large_patches'
    args.style_image = '/home/wcd/LinkToMyLib/Datas/BMVC_large_patches/0022832_blur.png'
    args.epochs = 2
    args.batch_size = 8
    args.lr = 1e-3
    args.content_weight = 1e4
    args.style_weight = 1e10
    args.checkpoint_model_dir = "ckpt"
    args.checkpoint_interval = 2000
    args.log_interval = 200 / args.batch_size
    args.plot_interval = 32
    args.seed = 42
    args.image_size = 128
    args.style_size = None
    args.HR_size = 128
    args.LR_scale = 1
    args.motion_len = 0
    args.motion_angel = 0
    args.gauss = 0
    args.mode = 'L'
    args.normalizebatch_flag = False
    args.deconv_mode = "US"
    args.resblobk_num = 8
    args.lossfun = "l1"
    args.model_name = 'TEXTnet'


    # args.subcommand = "eval"
    args.model = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/neural_style/ckpt/ckpt_epoch_0_batch_id_6000.pth"
    args.content_image = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/images/style-images/flowers.bmp"
    args.content_scale = 1
    args.output_image = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/images/output-images/flowers-6000.jpg"
    torch.backends.cudnn.benchmark = True

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
