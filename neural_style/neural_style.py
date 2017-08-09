import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms

import utils
from LRNet import TransformerNet
from vgg import Vgg16
import visdom


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
        transforms.Lambda(lambda x: x.mul(255))
    ])


    #load training folder
    train_dataset = utils.trainingFolder(args.dataset,args.HR_size,LR_scale=args.LR_scale,
                                         motion_len=40,motion_angel=20,gauss=3,
                                         transform=transform,target_transform=transform,
                                         )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transformer = TransformerNet()
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False)
    #load style image, which is not from dataset
    style = utils.load_LR_image(args.style_image, motion_len=40,motion_angel=20,gauss=3,
                                size=args.HR_size, scale=args.LR_scale, RGB=False)
    style = transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1)

    if args.cuda:
        transformer.cuda()
        vgg.cuda()
        style = style.cuda()

    style_v = Variable(style)
    style_v = utils.normalize_batch(style_v)
    features_style = vgg(style_v)
    gram_style = [utils.gram_matrix(y) for y in features_style]

    viz = visdom.Visdom()
    for e in range(args.epochs):

        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_ref_diff = 0.
        count = 0
        loss_win = None
        #load x, which is from train_loader
        for batch_id, (x, x_LR) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            if args.cuda:
                x, x_LR = x.cuda(), x_LR.cuda()
            x, x_LR = Variable(x),Variable(x_LR)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            y_LR = F.avg_pool2d(y,args.LR_scale)
            ref_diff = mse_loss(y_LR,x_LR)

            agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]
            agg_ref_diff += ref_diff.data[0]

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}\tref_diff: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1),
                                  agg_ref_diff /(batch_id + 1)
                )
                print(mesg)

            if (batch_id + 1) % args.plot_interval == 0:
                if loss_win is None :
                    loss_win = viz.line(
                        X=np.column_stack((batch_id, batch_id, batch_id)),
                        Y=np.column_stack((content_loss.data[0], style_loss.data[0], ref_diff.data[0])),
                        opts=dict(
                            markers=False,
                            legend=['content', 'style', 'ref'],
                        ),
                    )
                else:
                    viz.line(
                        # must specify x-values for line update
                        X=np.column_stack((batch_id, batch_id, batch_id)),
                        Y=np.column_stack((content_loss.data[0], style_loss.data[0], ref_diff.data[0])),
                        # 制定了update源是win这个pane
                        win=loss_win,
                        update='append',
                    )



            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval()
                if args.cuda:
                    transformer.cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                if args.cuda:
                    transformer.cuda()
                transformer.train()

    # save model
    transformer.eval()
    if args.cuda:
        transformer.cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    content_image = utils.load_HR_image(args.content_image,size=args.HR_size)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
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
    args.style_image = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/images/style-images/zebra.bmp"
    args.epochs = 2
    args.batch_size = 2
    args.lr = 1e-3
    args.content_weight = 1e5
    args.style_weight = 1e10
    args.checkpoint_model_dir = "ckpt"
    args.checkpoint_interval = 2000
    args.log_interval = 500
    args.plot_interval = 10
    args.seed = 42
    args.image_size = 128
    args.style_size = None
    args.HR_size = 128
    args.LR_scale = 1

    #args.subcommand = "eval"
    args.model = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/neural_style/ckpt/ckpt_epoch_0_batch_id_6000.pth"
    args.content_image = "/home/wcd/Projects/Pytorch-examples/fast_neural_style/images/style-images/flowers.bmp"
    args.content_scale = 2
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
