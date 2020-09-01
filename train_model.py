import argparse
import itertools
import numpy as np
import torchvision.transforms as transforms
import torchvision.utils
from torch.autograd import Variable
from PIL import Image
from matplotlib import pyplot as plt
import scipy.misc
import torch
import random
from unet_sftargmx import UNet
from unet_sftargmx import AttU_Net
from unet_sftargmx import Discriminator
# from models import Generator
# from models import Discriminator
# from models import UNet
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
import cv2
# from aggregate_ds_patch_based import *
# from aggregate_ds import *
# from aggregate_ds_unpaired import *
# import aggregate_point as agrd

import aggregate_point_set1 as agrd1
import aggregate_point_sorted as agrd

'''
training the model consist of 3 stages of stopping and resuming the process, 
this helps reinitazilizing the learning rate.
a rule of thumb: every 50 epochs one stop.
'''
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
# parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--sizex', type=int, default=400, help='size of the data crop (squared assumed)')
parser.add_argument('--sizey', type=int, default=100, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=25, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', default=True, help='use GPU computation')

opt = parser.parse_args()
torch.cuda.set_device(0) #gpu id
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
# netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_A2B = UNet(opt.input_nc, opt.output_nc)
# netG_A2B = AttU_Net(opt.input_nc, opt.output_nc)

netD_A = Discriminator(opt.input_nc)
# netD_B = Discriminator(opt.output_nc)

# if opt.cuda:
if True:
    netG_A2B.cuda()
    # netG_B2A.cuda()
    # netD_A.cuda()
    # netD_B.cuda()

netG_A2B.apply(weights_init_normal)
# netG_B2A.apply(weights_init_normal)
# netD_A.apply(weights_init_normal)
# netD_B.apply(weights_init_normal)

# netG_A2B.load_state_dict(torch.load('./output/models/unet_sag_point_set0.pth'))
# netG_A2B.load_state_dict(torch.load('./output/models/unet_sag_point_working3.pth'))
netG_A2B.load_state_dict(torch.load('./output/models/unet_sag_point_set1.pth'))
netG_A2B.load_state_dict(torch.load('./output/models/unet_sag_point_corrected.pth'))

netG_A2B.train()

# Lossess
criterion_bce = torch.nn.BCEWithLogitsLoss(size_average=False)
criterion_bce_2 = torch.nn.BCEWithLogitsLoss(reduce=False)
# criterion_bce_2 = torch.nn.BCEWithLogitsLoss()
criterion_GAN = torch.nn.BCEWithLogitsLoss()#torch.nn.MSELoss()
# criterion_cycle = torch.nn.BCEWithLogitsLoss()
criterion_mse = torch.nn.MSELoss(size_average=False)
criterion_mse2 = torch.nn.MSELoss()
# criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_huber = torch.nn.SmoothL1Loss()

# Optimizers & LR schedulers
# optimizer_G = torch.optim.Adam(netG_A2B.parameters())
# optimizer_G = torch.optim.RMSprop(netG_A2B.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.9, centered=False)
# note: try adam with , eps=1e-04
optimizer_G = torch.optim.Adam(netG_A2B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
# optimizer_D_B = torch.optim.RMSprop(netD_B.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.9, centered=False)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.sizex, opt.sizey)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.sizex, opt.sizey)
target_real = Variable(Tensor(opt.batchSize).unsqueeze(1).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).unsqueeze(1).fill_(0.0), requires_grad=False)

# fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

loss_G_list = []
loss_test_list = []
loss_D_list = []
loss_points_list = []
test_distance_list = []

# Loss plot
# logger = Logger(opt.n_epochs, len(radiographs))
###################################

# def save_output(real_A, fake_B, index, batchSize, epoch, test=False):
#     i = random.randint(0,batchSize-1)
#     i = int(batchSize/2)
#     # for i in range(batchSize):
#     real_A_tmp = real_A[i,:,:,:].squeeze().cpu()#.numpy()
#     if epoch == 0:
#         file_name = './output/images/real_A_{}_{}.jpg'.format(index, i)
#         if test:
#             file_name = './output/images/test/real_A_{}_{}.jpg'.format(index, i)
#         torchvision.utils.save_image(real_A_tmp, file_name)
#     if epoch%2 !=0:
#         return
#     fake_B_tmp = fake_B[i,:,:,:].cpu().data.numpy()
#     tmp = np.sum(fake_B_tmp, axis=0)#.data.numpy()
#     file_name = './output/images/fake_B_{}_{}.jpg'.format(index, i)
#     if test:
#         file_name = './output/images/test/fake_B_{}_{}.jpg'.format(index, i)
#     texted_image = np.copy(tmp)
#     for dim in range(fake_B_tmp.shape[0]-1):
#         ind = np.unravel_index(np.argmax(fake_B_tmp[dim,:,:], axis=None), fake_B_tmp[dim,:,:].shape)
#         texted_image =cv2.putText(img=texted_image, text=str(dim+1),org=(ind[1]+10,ind[0]),fontFace=1, fontScale=1, color=(1,1,1), thickness=2)
#         # texted_image =cv2.putText(img=texted_image, text="hello", org=(50, 50+dim*2),fontFace=1, fontScale=1, color=(1,1,1), thickness=2)
#     # plt.imshow(texted_image)
#     plt.imsave(file_name, texted_image)


def save_output(real_A, fake_B, points, offset, real_points, index, batchSize, epoch, test=False):
    print('real_A.shape, fake_B.shape', real_A.shape, fake_B.shape)
    i = random.randint(0,batchSize-1)
    i = int(batchSize/2)
    i = 0
    # for i in range(batchSize):
    real_A_tmp = real_A[i,:,:,:].squeeze().cpu()#.numpy()
    if epoch == 0:
        file_name = './output/images/real_A_{}_{}.jpg'.format(index, i)
        if test:
            file_name = './output/images/test/real_A_{}_{}.jpg'.format(index, i)
        torchvision.utils.save_image(real_A_tmp, file_name)
    if epoch%2 !=0:
        return
    real_A_tmp = real_A_tmp.numpy()
    fake_B_tmp = fake_B[i,:,:,:].cpu().data.numpy()
    points_tmp = points[i,:,:].cpu().data.numpy()
    real_points_tmp = real_points[i,:,:].cpu().data.numpy()
    # print points_tmp
    points_tmp = points_tmp.astype(int)
    real_points_tmp = real_points_tmp.astype(int)
    # print points_tmp
    tmp = np.sum(fake_B_tmp, axis=0)#.data.numpy()
    file_name = './output/images/fake_B_{}_{}.jpg'.format(index, i)
    file_name2 = './output/images/real_overlay_{}_{}.jpg'.format(index, i)
    if test:
        file_name = './output/images/test/fake_B_{}_{}.jpg'.format(index, i)
        file_name2 = './output/images/test/real_overlay_{}_{}.jpg'.format(index, i)
    texted_image = np.copy(tmp)
    for dim in range(points_tmp.shape[0]):
        # ind = np.unravel_index(np.argmax(fake_B_tmp[dim,:,:], axis=None), fake_B_tmp[dim,:,:].shape)
        ind = points_tmp[dim, :]
        real_ind = real_points_tmp[dim, :]
        texted_image =cv2.putText(img=texted_image, text=str(dim+1),org=(ind[1]+10,ind[0]),fontFace=1, fontScale=1, color=(1,1,1), thickness=2)
        texted_image =cv2.putText(img=texted_image, text=str(dim+1),org=(real_ind[1]+20,real_ind[0]),fontFace=1, fontScale=1, color=(5,1,2), thickness=2)
        real_A_tmp =cv2.putText(img=real_A_tmp, text=str(dim+1),org=(ind[1]+10,ind[0]),fontFace=1, fontScale=1, color=(1,1,1), thickness=2)
        real_A_tmp =cv2.putText(img=real_A_tmp, text=str(dim+1),org=(real_ind[1]+20,real_ind[0]),fontFace=1, fontScale=1, color=(5,1,2), thickness=2)
        # texted_image =cv2.putText(img=texted_image, text="hello", org=(50, 50+dim*2),fontFace=1, fontScale=1, color=(1,1,1), thickness=2)
    # plt.imshow(texted_image)
    plt.imsave(file_name, texted_image)
    plt.imsave(file_name2, real_A_tmp)

# def save_output(real_A, real_B, fake_B, index, batchSize, epoch, test=False):
#     i = random.randint(0,batchSize-1)
#     i = int(batchSize/2)
#     # for i in range(batchSize):
#     real_A_tmp = real_A[i,:,:,:].squeeze().cpu()#.numpy()
#     if epoch == 0:
#         file_name = './output/images/real_A_{}_{}.jpg'.format(index, i)
#         if test:
#             file_name = './output/images/test/real_A_{}_{}.jpg'.format(index, i)
#         # print file_name
#         torchvision.utils.save_image(real_A_tmp, file_name)

#         real_B_tmp = real_B[i,:,:,:].squeeze().cpu()#.numpy()

#         tmp = torch.sum(real_B_tmp, dim=0)

#         file_name = './output/images/real_B_{}_{}.jpg'.format(index, i)
#         if test:
#             file_name = './output/images/test/real_B_{}_{}.jpg'.format(index, i)
#         # scipy.misc.imsave(file_name, tmp.data.numpy())
#         # torchvision.utils.save_image(tmp, file_name)
#         texted_image = np.copy(tmp)
#         for dim in range(real_B_tmp.shape[0]-1):
#             ind = np.unravel_index(np.argmax(real_B_tmp[dim,:,:], axis=None), real_B_tmp[dim,:,:].shape)
#             texted_image =cv2.putText(img=texted_image, text=str(dim+1),org=(ind[1]+10,ind[0]),fontFace=1, fontScale=1, color=(1,1,1), thickness=2)

#         plt.imsave(file_name, texted_image)

#     fake_B_tmp = fake_B[i,:,:,:].cpu().data.numpy()
#     tmp = np.sum(fake_B_tmp, axis=0)#.data.numpy()
#     file_name = './output/images/fake_B_{}_{}.jpg'.format(index, i)
#     if test:
#         file_name = './output/images/test/fake_B_{}_{}.jpg'.format(index, i)
#     texted_image = np.copy(tmp)
#     for dim in range(fake_B_tmp.shape[0]-1):
#         ind = np.unravel_index(np.argmax(fake_B_tmp[dim,:,:], axis=None), fake_B_tmp[dim,:,:].shape)
#         texted_image =cv2.putText(img=texted_image, text=str(dim+1),org=(ind[1]+10,ind[0]),fontFace=1, fontScale=1, color=(1,1,1), thickness=2)
#         # texted_image =cv2.putText(img=texted_image, text="hello", org=(50, 50+dim*2),fontFace=1, fontScale=1, color=(1,1,1), thickness=2)
#     # plt.imshow(texted_image)
#     plt.imsave(file_name, texted_image)


# x_train, y_train, point_train = agrd_fast.get_data()
# x_test, y_train, point_test = agrd_fast.get_data(test=True)
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    # for i, batch in enumerate(datast):
    # if epoch == 150:
    #     break
    i = 0
    netG_A2B.train()
    for img, seg, target_points in agrd.get_data():
    # for item in zip(x_train, y_train, point_train):
        # Set model input
        # img, seg, target_points = item[0], item[1], item[2]
        image_A = torch.from_numpy(img).unsqueeze(1).float()
        image_B = torch.from_numpy(seg).float()

        image_A = image_A#/torch.max(image_A)
        real_A = Variable(image_A).cuda()
        real_B = Variable(image_B).cuda()
        real_points = Variable(torch.from_numpy(target_points).float()).cuda()

        # ###### Generators A2B and B2A ######
        # # GAN loss
        # fake_B = netG_A2B(real_A)
        # print fake_B.shape
        # print real_A.shape
        # print real_B.shape
        # print fake_B
        fake_B, points = netG_A2B(real_A)
        # print points.shape
        # print points.view(points.size()[0], -1).shape
        # print real_p.shape
        # print real_p.view(points.size()[0], -1).shape

        diffX = real_A.size()[2] - fake_B.size()[2]
        diffY = real_A.size()[3] - fake_B.size()[3]

        # apply the shift on predicte points or the target points

        ###################################

        # ##### Discriminator B ######
        # optimizer_D_B.zero_grad()

        # # Real loss
        # # pred_real = netD_B(real_B[:,:,diffX//2:fake_B.shape[2]+diffX//2,diffY//2:fake_B.shape[3]+diffY//2])
        # pred_real = netD_B(real_points.view(1,-1))
        # loss_D_real = criterion_GAN(pred_real, target_real)
        
        # # Fake loss
        # # fake_B = fake_B_buffer.push_and_pop(fake_B.detach())
        # # pred_fake = netD_B(fake_B.detach())
        # pred_fake = netD_B(points.view(1,-1))
        # loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # # Total loss
        # loss_D_B = (loss_D_real + loss_D_fake)*100
        # loss_D_B.backward()
        # # print 'loss D B: {}'.format(loss_D_B)

        # optimizer_D_B.step()
        ###################################

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # GAN loss
        # pred_fake = netD_B(fake_B)
        # loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        bce_loss = criterion_bce_2(fake_B, real_B[:,:,diffX//2:fake_B.shape[2]+diffX//2,diffY//2:fake_B.shape[3]+diffY//2])
        one_hot = torch.zeros_like(bce_loss).cuda()
        one_hot.scatter_(1, torch.argmax(bce_loss, dim=1).unsqueeze(1), 1)
        
        loss_generator = criterion_mse(fake_B, real_B[:,:,diffX//2:fake_B.shape[2]+diffX//2,diffY//2:fake_B.shape[3]+diffY//2])
        loss_generator += torch.sum(one_hot*criterion_bce_2(fake_B, real_B[:,:,diffX//2:fake_B.shape[2]+diffX//2,diffY//2:fake_B.shape[3]+diffY//2]))

        offset = torch.zeros_like(points[:,:,:]).cuda()
        offset[:,:,0] = diffX//2
        offset[:,:,1] = diffY//2
        points += offset
        # loss_point = criterion_mse2(points[:,:-1,:], real_points)
        loss_point = criterion_huber(points[:,:-1,:], real_points)
        # loss_generator += loss_point#*0.0001
        total_loss = loss_point + loss_generator
        #test_distance = torch.mean(torch.sqrt(torch.mul((points[:,:-1,:] - real_points), (points[:,:-1,:] - real_points))))
        #print('test_distance',test_distance)
        # print 'max: {}'.format(torch.max(fake_B))
        # print 'min: {}'.format(torch.min(fake_B))
        # print 'max gt: {}'.format(torch.max(real_B))
        # print 'min gt: {}'.format(torch.min(real_B))
        # loss_generator += criterion_identity(d1, real_B[:,:,diffX//2:fake_B.shape[2]+diffX//2,diffY//2:fake_B.shape[3]+diffY//2])
        # loss_generator += criterion_identity(d2, real_B[:,:,diffX//2:fake_B.shape[2]+diffX//2,diffY//2:fake_B.shape[3]+diffY//2])
        # loss_generator += criterion_identity(d3, real_B[:,:,diffX//2:fake_B.shape[2]+diffX//2,diffY//2:fake_B.shape[3]+diffY//2])
        # loss_generator += criterion_identity(d4, real_B[:,:,diffX//2:fake_B.shape[2]+diffX//2,diffY//2:fake_B.shape[3]+diffY//2])
        # loss_generator += loss_GAN_A2B
        total_loss.backward()
        
        optimizer_G.step()
        ###################################

        # loss_G_list.append(loss_generator.detach().cpu())
        loss_points_list.append(loss_point.detach().cpu())
        # loss_D_list.append(loss_D_B)
        # loss_shape = criterion_mse2(fake_B, real_B[:,:,diffX//2:fake_B.shape[2]+diffX//2,diffY//2:fake_B.shape[3]+diffY//2])
        # loss_shape = torch.abs(fake_B - real_B[:,:,diffX//2:fake_B.shape[2]+diffX//2,diffY//2:fake_B.shape[3]+diffY//2])
        # print 'max loss: {}'.format(torch.max(loss_shape))
        # print 'min loss: {}'.format(torch.min(loss_shape))
        # print '-------'*3
        save_output(real_A, fake_B, points[:,:-1,:], offset, real_points ,i, opt.batchSize, epoch)
        # save_output(real_A, real_B, fake_B, i, opt.batchSize, epoch)
        i += 1
        # torch.cuda.empty_cache()
        # break
    i = 0
    lr_scheduler_G.step()
    # torch.save(netG_A2B.state_dict(), 'output/models/unet_sag_point_set1.pth')
    torch.save(netG_A2B.state_dict(), 'output/models/unet_sag_point_corrected.pth')
    # print 'test...'
    netG_A2B.eval()
    for img, seg, target_points in agrd1.get_data(test=True):
    # for item in zip(x_test, point_test):
        # img, target_points = item[0], item[1]
        with torch.no_grad():
            # print 'beggining'
            image_A = torch.from_numpy(img).unsqueeze(1).float()
            # image_B = torch.from_numpy(seg).float()
            
            real_A = Variable(image_A).cuda()
            real_B = Variable(image_B).cuda()
            real_points = Variable(torch.from_numpy(target_points).float()).cuda()

            fake_B, points = netG_A2B(real_A.detach())
            diffX = real_A.size()[2] - fake_B.size()[2]
            diffY = real_A.size()[3] - fake_B.size()[3]
            # print 'output'
            ###### Generators A2B and B2A ######
            # bce_loss = criterion_bce_2(fake_B, real_B[:,:,diffX//2:fake_B.shape[2]+diffX//2,diffY//2:fake_B.shape[3]+diffY//2])
            # one_hot = torch.zeros_like(bce_loss).cuda()
            # one_hot.scatter_(1, torch.argmax(bce_loss, dim=1).unsqueeze(1), 1)
            
            # loss_generator = criterion_mse(fake_B, real_B[:,:,diffX//2:fake_B.shape[2]+diffX//2,diffY//2:fake_B.shape[3]+diffY//2])
            # loss_generator += torch.sum(one_hot*criterion_bce_2(fake_B, real_B[:,:,diffX//2:fake_B.shape[2]+diffX//2,diffY//2:fake_B.shape[3]+diffY//2]))

            offset = torch.zeros_like(points[:,:,:]).cuda()
            offset[:,:,0] = diffX//2
            offset[:,:,1] = diffY//2
            points += offset
            # print 'loss:'
            loss_point = criterion_huber(points[:,:-1,:], real_points)
            loss_generator = loss_point
            loss_test_list.append(loss_generator.detach().cpu())
            # test_distance = torch.sqrt(torch.sum(torch.pow((points[:,:-1,:] - real_points), 2), axis=2))
            #test_distance = torch.sqrt(criterion_mse2(points[:,:-1,:], real_points))
            test_distance = torch.mean(torch.sqrt(torch.mul((points[:,:-1,:] - real_points), (points[:,:-1,:] - real_points))))
            # print('test_distance:', test_distance)
            test_distance_list.append(test_distance)
            save_output(real_A, fake_B, points[:,:-1,:], offset, real_points, i, opt.batchSize, epoch, test=True)
            # save_output(real_A, real_B, fake_B, i, opt.batchSize, epoch, test=True)
            i += 1
            # torch.cuda.empty_cache()
        # print ok
    # if epoch % 19 == 0:
    
    print 'end of epoch'
    print 'test point loss: {}'.format(torch.mean(torch.stack(loss_test_list)))
    print 'test test_distance: {}'.format(torch.mean(torch.stack(test_distance_list)))
    # print 'generator loss: {}'.format(torch.mean(torch.stack(loss_G_list)))
    print 'point loss: {}'.format(torch.mean(torch.stack(loss_points_list)))
    print 'epoch no: {}'.format(epoch)
    print '----------------'*4
    
