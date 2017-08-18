from PIL import Image
import  numpy as np
import visdom
def drawloss(viz,loss_win,batch_id,loss):
    if loss_win is None:
        loss_win = viz.line(
            # X=np.column_stack((batch_id, batch_id, batch_id)),
            X=np.column_stack((batch_id, batch_id)),
            Y=np.column_stack(loss),
            opts=dict(
                markers=False,
                legend=['train', 'val'],
            ),
        )
    else:
        loss_win = viz.line(
            # must specify x_HR-values for line update
            X=np.column_stack((batch_id, batch_id)),
            Y=np.column_stack(loss),
            # 制定了update源是win这个pane
            win=loss_win,
            update='append',
            opts=dict(
                markers=False,
                legend=['train', 'val'],
            ),
        )
    return loss_win

def drawdifloss(viz,loss_win,batch_id,loss,legend):
    if loss_win is None:
        loss_win = viz.line(
            # X=np.column_stack((batch_id, batch_id, batch_id)),
            X=np.column_stack((batch_id, batch_id)),
            Y=np.column_stack(loss),
            opts=dict(
                markers=False,
                legend=legend,
            ),
        )
    else:
        loss_win = viz.line(
            # must specify x_HR-values for line update
            X=np.column_stack((batch_id, batch_id)),
            Y=np.column_stack(loss),
            # 制定了update源是win这个pane
            win=loss_win,
            update='append',
            opts=dict(
                markers=False,
                legend=legend,
            ),
        )
    return loss_win

def var2imgnp(var):
    #input: Variable of size 1xcxhxw value(0,255+)
    #output: numpy of size cxhxw value(0,255)
    return var.data.cpu().squeeze(0).clamp(0, 255).numpy()

def imgnp_convert(img,mode='RGB',Cb=None,Cr=None):
    #input: numpy of size cxhxw,image mode( when display the img, need to change to RGB)
    #if the origin mode is Y, Cb Cr is also needed to display a merge img
    #output: numpy of size cxhxw
    if mode is 'YCbCr':
        img = img.transpose(1, 2, 0).astype("uint8")
        Y = Image.fromarray(img[:, :, 0], 'L')
        Cb = Image.fromarray(img[:, :, 1], 'L')
        Cr = Image.fromarray(img[:, :, 2], 'L')
        img = Image.merge('YCbCr', [Y, Cb, Cr]).convert('RGB')
        return np.array(img).transpose(2, 0, 1)
    elif mode is 'Y':
        img = img.astype("uint8")
        img = Image.fromarray(img[0],'L')
        img = Image.merge('YCbCr', [img, Cb, Cr]).convert('RGB')
        return np.array(img).transpose(2, 0, 1)
    elif mode is 'L':
        return img[0].astype("uint8")
    elif mode is 'RGB':
        return img.astype("uint8")
    else:
        print('error: imgnp_convert mode undifine')

import numpy as np
import time

class Visualizer():
    def __init__(self):
        self.vis = visdom.Visdom(env='wcd')
        self.display_id = 1

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals):

        idx = 1
        for label, image_numpy in visuals.items():
            self.vis.image(image_numpy.transpose([2,0,1]), opts=dict(title=label),
                               win=self.display_id + idx)
            idx += 1

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


