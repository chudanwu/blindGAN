from PIL import Image
import  numpy as np

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