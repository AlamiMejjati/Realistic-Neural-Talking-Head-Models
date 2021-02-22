import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib

import numpy as np

from dataset.dataset_class import FineTuningImagesDataset, FineTuningVideoDataset
from network.model import *
from loss.loss_discriminator import *
from loss.loss_generator import *
from torch.utils.tensorboard import SummaryWriter
from params.params import path_to_Wi

from utils import imsaver

import os

"""Hyperparameters and config"""
display_training = True
if not display_training:
	matplotlib.use('agg')
device = torch.device("cuda:0")
cpu = torch.device("cpu")
path_to_chkpt = '/home/youssef/Documents/phdYoop/Realistic-Neural-Talking-Head-Models/' \
                'train_log/2021-02-18_20-42-21/model_weights.tar'
save_path = '/media/youssef/SSD2/phdYoop/Realistic-Neural-Talking-Head-Models/personalData/paul'

path_to_save = os.path.join(save_path, 'models', 'finetuned_model.tar')
path_to_video = os.path.join(save_path, 'vid', '00059.mp4')
path_to_images = os.path.join(save_path, 'ims')


"""Create dataset and net"""
choice = ''
while choice != '0' and choice != '1':
    choice = input('What source to finetune on?\n0: Video\n1: Images\n\nEnter number\n>>')

if choice == '0': #video
    dataset = FineTuningVideoDataset(path_to_video, device)
    path_to_embedding = os.path.join(save_path, 'latent', 'e_hat_video.tar')
else: #Images
    dataset = FineTuningImagesDataset(path_to_images, device)
    path_to_embedding = os.path.join(save_path, 'latent', 'e_hat_images.tar')

dataLoader = DataLoader(dataset, batch_size=2, shuffle=False)

e_hat = torch.load(path_to_embedding, map_location=cpu)
e_hat = e_hat['e_hat']

G = Generator(256, finetuning=True, e_finetuning=e_hat)
D = Discriminator(dataset.__len__(), path_to_Wi, finetuning=True, e_finetuning=e_hat)

G.train()
D.train()

optimizerG = optim.Adam(params = G.parameters(), lr=5e-5)
optimizerD = optim.Adam(params = D.parameters(), lr=2e-4)


"""Criterion"""
criterionG = LossGF(VGGFace_body_path='Pytorch_VGGFACE_IR.py',
                   VGGFace_weight_path='Pytorch_VGGFACE.pth', device=device)
criterionDreal = LossDSCreal()
criterionDfake = LossDSCfake()


"""Training init"""
epochCurrent = epoch = i_batch = 0
lossesG = []
lossesD = []
i_batch_current = 0

num_epochs = 40

#Warning if checkpoint inexistant
if not os.path.isfile(path_to_chkpt):
    print('ERROR: cannot find checkpoint')
if os.path.isfile(path_to_save):
    path_to_chkpt = path_to_save

"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
checkpoint['D_state_dict']['W_i'] = torch.rand(512, 32) #change W_i for finetuning

G.load_state_dict(checkpoint['G_state_dict'])
D.load_state_dict(checkpoint['D_state_dict'], strict = False)


"""Change to finetuning mode"""
G.finetuning_init()
D.finetuning_init()

G.to(device)
D.to(device)

"""Training"""
batch_start = datetime.now()

cont = True
counter = 0
writer = SummaryWriter(log_dir=os.path.join(save_path, 'models'))
while cont:
    for epoch in range(num_epochs):
        for i_batch, (x, g_y) in enumerate(dataLoader):
            with torch.autograd.enable_grad():
                #zero the parameter gradients
                optimizerG.zero_grad()
                optimizerD.zero_grad()
    
                #forward
                #train G and D
                x_hat = G(g_y, e_hat)
                r_hat, D_hat_res_list = D(x_hat, g_y, i=0)
                with torch.no_grad():
                    r, D_res_list = D(x, g_y, i=0)
    
                dict_loss_G = criterionG(x, x_hat, r_hat, D_res_list, D_hat_res_list)
                lossG = sum(dict_loss_G.values())
                lossG.backward(retain_graph=False)
                optimizerG.step()

                #train D
                optimizerD.zero_grad()
                x_hat.detach_().requires_grad_()
                r_hat, D_hat_res_list = D(x_hat, g_y, i=0)
                r, D_res_list = D(x, g_y, i=0)
    
                lossDfake = criterionDfake(r_hat)
                lossDreal = criterionDreal(r)
    
                lossD = lossDreal + lossDfake
                lossD.backward(retain_graph=False)
                optimizerD.step()
                
                
                #train D again
                # optimizerG.zero_grad()
                # optimizerD.zero_grad()
                # r_hat, D_hat_res_list = D(x_hat, g_y, i=0)
                # r, D_res_list = D(x, g_y, i=0)
                #
                # lossDfake = criterionDfake(r_hat)
                # lossDreal = criterionDreal(r)
                #
                # lossD = lossDreal + lossDfake
                # lossD.backward(retain_graph=False)
                # optimizerD.step()
    
    
            # Output training stats
            if epoch % 10 == 0:
                batch_end = datetime.now()
                avg_time = (batch_end - batch_start) / 10
                print('\n\navg batch time for batch size of', x.shape[0],':',avg_time)
                
                batch_start = datetime.now()
                
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(y)): %.4f'
                      % (epoch, num_epochs, i_batch, len(dataLoader),
                         lossD.item(), lossG.item(), r.mean(), r_hat.mean()))

                imsaver(x, x_hat, g_y, counter, writer)
                counter+=1

    num_epochs = int(input('Num epoch further?\n'))
    cont = num_epochs != 0

print('Saving finetuned model...')
torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'lossesD': lossesD,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        }, path_to_save)
print('...Done saving latest')
