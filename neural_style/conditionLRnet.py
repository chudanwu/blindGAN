import argparse
import os
import sys
import time
import numpy as np
import math
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import utils
import draw
from LRnet import condition_lrNet,create_condition
import visdom
from models import crossEntropy2d
from dataFolder import randomBlurFolder

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
        torch.cuda.set_device(1)
        torch.cuda.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x_HR: x_HR.mul(255))
    ])

    # load training folder
    train_dataset = randomBlurFolder(args.dataset, args.HR_size, LR_scale=None,
                                         motion_len_max=args.motion_len_max, motion_angel_max=args.motion_angel_max, gauss_max=args.gauss_max,
                                         transform=transform, target_transform=transform, mode=args.mode)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    # load validate folder
    val_dataset = randomBlurFolder(args.valset, args.HR_size, LR_scale=None,
                                   motion_len_max=args.motion_len_max, motion_angel_max=args.motion_angel_max,
                                   gauss_max=args.gauss_max,
                                       transform=transform, target_transform=transform, mode=args.mode)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    transformer = condition_lrNet(args.mode, resblock_num=args.resblobk_num, channelofparams=3)
    if args.cuda:
        transformer = transformer.cuda()
    optimizer = Adam(transformer.parameters(), args.lr)

    # define loss function:
    if args.lossfun is "mse":
        # 数量级：1e2
        lossfun = torch.nn.MSELoss()
    elif args.lossfun is "l1":
        # 数量级：1e1
        lossfun = torch.nn.L1Loss()
    else:
        lossfun = torch.nn.L1Loss()

    # 数量级：1e2~1e3
    tvloss = utils.TVLoss()
    l1reg = utils.L1Reg()
    l1edgereg = utils.L1edgeReg()
    nll = crossEntropy2d()
    mse = torch.nn.MSELoss()


    # 读取HR用来测试和看
    if args.mode is 'Y' :
        ref_LR_y, ref_LR_b, ref_LR_r = utils.load_LR_image(args.style_image,
                                                           motion_len=args.motion_len, motion_angel=args.motion_angel,
                                                           gauss=args.gauss,
                                                           scale=None, mode='YCbCr').split()
        ref_LR = ref_LR_y
    else:
        ref_LR = utils.load_LR_image(args.style_image,
                                     motion_len=args.motion_len, motion_angel=args.motion_angel,
                                     gauss=args.gauss,
                                     scale=None, mode=args.mode)
        ref_LR_b = None
        ref_LR_r = None

    # 用来测试的HR

    viz = visdom.Visdom(env='wcd')

    for e in range(args.epochs):

        transformer.train()
        agg_content_loss = 0.
        count = 0
        loss_win = None
        bestloss = None

        for batch_id, (x_HR, x_LR, x_params) in enumerate(train_loader):
            n_batch = len(x_HR)
            count += n_batch
            optimizer.zero_grad()
            x_condition = create_condition(n_batch, x_params, args.HR_size, args.HR_size, gauss_max=args.gauss_max,
                                           motion_x_max=args.motion_len_max, motion_y_max=args.motion_len)
            if args.cuda:
                x_LR ,x_condition= x_LR.cuda(),x_condition.cuda()
            x_LR ,x_condition= Variable(x_LR), Variable(x_condition)

            x_p = transformer(x_LR)
            content_loss = args.content_weight * lossfun(x_p, x_condition)

            content_loss.backward()
            optimizer.step()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = (
                "{}\tEpoach {}:\t[{}/{}]\t" + args.lossfun + "loss: {:.6f}\t").format(
                    time.ctime(), e + 1, count, len(train_dataset), content_loss.data[0] / n_batch
                )
                print(mesg)

            if (batch_id + 1) % args.plot_interval == 0:
                val_loss = validate(args, transformer, val_loader=val_loader, loss_function=lossfun)
                loss = (content_loss.data[0] / n_batch, val_loss)
                loss_win = draw.drawloss(viz, loss_win, batch_id, loss)
                if bestloss is None:
                    bestloss = val_loss
                elif bestloss > val_loss and (e > 1 or batch_id > 500):
                    transformer.eval()
                    bestloss = val_loss
                    if args.cuda:
                        transformer.cpu()
                    utils.save_model(transformer, args, bestloss, e, batch_id)
                    if args.cuda:
                        transformer.cuda()
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

    style_model = condition_TransformerNet()
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
    for batch_id, (v_HR, v_LR, v_param) in enumerate(val_loader):
        n_batch = len(v_HR)
        v_condition = create_condition(n_batch,v_param, args.HR_size,args.HR_size,gauss_max=args.gauss_max,
                                           motion_x_max=args.motion_len_max, motion_y_max=args.motion_len)
        count += n_batch
        if args.cuda:
            v_HR, v_LR ,v_condition= v_HR.cuda(), v_LR.cuda(), v_condition.cuda()
        v_HR, v_LR ,v_condition= Variable(v_HR, requires_grad=False), Variable(v_LR, requires_grad=False),Variable(v_condition,requires_grad=False)

        v_p = net(v_LR)

        content_loss = args.content_weight * loss_function(v_p,v_condition)
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
    args.dataset = '/home/wcd/LinkToMyLib/Datas/train2014'
    args.valset = "/media/library/wcd/Datas/BSDS/BSDS300/images/train"
    args.style_image = "/home/wcd/Desktop/text.jpg"
    args.epochs = 5
    args.batch_size = 8
    args.lr = 1e-4
    args.content_weight = 1
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
    args.motion_len = 5
    args.motion_angel = 33
    args.motion_x = args.motion_len*math.cos(math.radians(args.motion_angel))
    args.motion_y = args.motion_len*math.sin(math.radians(args.motion_angel))
    args.gauss = 2.5
    args.gauss_max = 2.5
    args.motion_len_max = 7
    args.motion_angel_max = 90
    args.mode = 'L'
    args.normalizebatch_flag = False
    args.deconv_mode = "US"
    args.resblobk_num = 12
    args.lossfun = "mse"
    args.model_name = 'condition_LRnet'


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
