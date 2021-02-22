import torchvision
import torch


def getIms(x, x_hat, g_y, f_lm):
    outx = (x_hat * 255)
    landmarks = (g_y * 255)
    gtx = x * 255.
    composite = getComposite(outx, landmarks)

    identity = f_lm[:,0,:,:,:,:]
    identity_rgb = (identity[:,0,:,:,:] * 255)
    identity_landmarks = (identity[:,1,:,:,:] * 255)
    identity_composite = getComposite(identity_rgb, identity_landmarks)

    imout = torch.cat([identity_rgb, identity_composite, gtx, composite, outx], dim=2).transpose(2,3)/255.
    # grid = torchvision.utils.make_grid(imout)
    return imout

def getIms_ft(x, x_hat, g_y):
    outx = (x_hat * 255)
    landmarks = (g_y * 255)
    gtx = x * 255.
    composite = getComposite(outx, landmarks)


    imout = torch.cat([gtx, composite, outx], dim=2).transpose(2,3)/255.
    # grid = torchvision.utils.make_grid(imout)
    return imout

def getComposite(outx, landmarks):
    bin_landmarks = 255. - torch.mean(landmarks, dim=1, keepdim=True)
    bin_landmarks = (bin_landmarks > 0.) * 1.
    composite = outx * (1 - bin_landmarks) + landmarks * bin_landmarks
    return composite

def saver(epoch, lossesG, lossesD, E, D, G, dataset, i_batch, optimizerG,
          optimizerD, path_to_chkpt, x, x_hat, g_y, f_lm, counter, writer):
    print('Saving latest...')
    torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'lossesD': lossesD,
        'E_state_dict': E.module.state_dict(),
        'G_state_dict': G.module.state_dict(),
        'D_state_dict': D.module.state_dict(),
        'num_vid': dataset.__len__(),
        'i_batch': i_batch,
        'optimizerG': optimizerG.state_dict(),
        'optimizerD': optimizerD.state_dict()
    }, path_to_chkpt)
    print('...Done saving latest')
    imgrid = getIms(x, x_hat, g_y, f_lm)
    [writer.add_image('images/%d' %k, imgrid[k], counter) for k in range(imgrid.shape[0])]

def imsaver(x, x_hat, g_y, counter, writer):
    imgrid = getIms_ft(x, x_hat, g_y)
    [writer.add_image('images/%d' %k, imgrid[k], counter) for k in range(imgrid.shape[0])]