import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
from unet_parts import *
from soft_argmax import SoftArgmax2D

'''
Attention U-Net code is borrowed from here:
https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
'''

mean_point =  [[  81.08733275,   65.03879815],
  [ 102.58776189,   65.03879815],
  [ 127.00610642,   69.10852223],
  [ 146.81721612,   74.944353  ],
  [ 167.16583656,   84.15882263],
  [ 185.97871206,   94.67867546],
  [ 206.0201835,   107.5021457 ],
  [ 226.06165495,  121.01670116],
  [ 247.2549351,   134.14732038],
  [ 268.37142801,  146.81721612],
  [ 292.02190006,  156.95313272],
  [ 315.82594661,  165.47651713],
  [ 341.0121636,  170.69804992],
  [ 368.27163626,  173.53917805],
  [ 394.53287471,  172.54094384],
  [ 421.48519838,  169.23909223],
  [ 450.58756496,  163.09611247],
  [ 482.45427244,  152.65304689],
  [ 515.54957586,  140.59744912],
  [ 551.02528394,  128.7722131 ],
  [ 589.18854567,  119.78810521],
  [ 626.81429666,  117.17733881],
  [ 659.6024511,   125.16321249],
  [ 687.0922855,   140.44387463]]
mean_point = np.asarray(mean_point)
mean_point = torch.from_numpy(mean_point).view(1,-1)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Linear(52, 30),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model = [   nn.Linear(30, 2),
                    # nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        # print x.shape
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512, dilation=2)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

        # self.dsn1 = nn.Conv2d(256, 1, 1)
        # self.dsn2 = nn.Conv2d(128, 1, 1)
        # self.dsn3 = nn.Conv2d(64, 1, 1)
        # self.dsn4 = nn.Conv2d(64, 1, 1)

        self.soft_argmax = SoftArgmax2D(window_fn="Uniform")
        # self.soft_argmax = SoftArgmax2D(window_fn="Parzen")
        model = [   nn.Linear(50, 50),
                    # nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=False) ]
        model += [   nn.Linear(50, 50),
                    # nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=False) ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x1 = self.inc(x)
        # print 'x1.shape: {}'.format(x1.shape)
        x2 = self.down1(x1)
        # print 'x2.shape: {}'.format(x2.shape)
        x3 = self.down2(x2)
        # print 'x3.shape: {}'.format(x3.shape)
        x4 = self.down3(x3)
        # print 'x4.shape: {}'.format(x4.shape)
        x5 = self.down4(x4)
        # print 'x5.shape: {}'.format(x5.shape)
        x1u = self.up1(x5, x4)
        # print 'x1.shape: {}'.format(x.shape)
        x2u = self.up2(x1u, x3)
        # print 'x2.shape: {}'.format(x.shape)
        x3u = self.up3(x2u, x2)
        # print 'x3.shape: {}'.format(x.shape)
        x4u = self.up4(x3u, x1)
        # print 'x4.shape: {}'.format(x.shape)
        x = self.outc(x4u)
        # print 'x1.shape: {}'.format(x.shape)
        # x = F.pad(x, (1, 1, 1, 1))
        h, w = x.shape[2], x.shape[3]
        # d1 = F.upsample_bilinear(self.dsn1(x1u), size=(h,w))
        # d2 = F.upsample_bilinear(self.dsn2(x2u), size=(h,w))
        # d3 = F.upsample_bilinear(self.dsn3(x3u), size=(h,w))
        # d4 = F.upsample_bilinear(self.dsn4(x4u), size=(h,w))
        # d1 = F.sigmoid(d1)
        # d2 = F.sigmoid(d2)
        # d3 = F.sigmoid(d3)
        # d4 = F.sigmoid(d4)
        heat_map = F.sigmoid(x)
        points_tmp = self.soft_argmax(heat_map[:,:,:,:])
        
        points = points_tmp.view(points_tmp.shape[0], -1)
        # print points.shape
        points = self.model(points)
        points = points.view(points.shape[0], 25, -1)
        points += points_tmp
        return heat_map, points



class AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=24):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024,dilation=1)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.soft_argmax = SoftArgmax2D(window_fn="Uniform")

        # correction = [
        #     nn.Conv2d(1,1,kernel_size=[25,2],padding=0),
        #     nn.LeakyReLU(0.2, inplace=True),]
        # correction2 = [
        #     nn.Conv2d(1,1,kernel_size=[25,2],padding=0),
        #     nn.LeakyReLU(0.2, inplace=True)
        # ]
        # self.correction = nn.Sequential(*correction)
        # self.correction2 = nn.Sequential(*correction2)
        self.linear1 = nn.Linear(50, 50, bias=True)
        self.linear1.bias.data = mean_point
        self.linear2 = nn.Linear(50, 50, bias=True)
        self.linear2.bias.data = mean_point
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=False)
        # model = [   nn.Linear(50, 50, bias=True),
        #             # nn.InstanceNorm2d(128), 
        #             nn.LeakyReLU(0.2, inplace=False) ]
        # model += [   nn.Linear(50, 50, bias=True),
        #             # nn.InstanceNorm2d(128), 
        #             nn.LeakyReLU(0.2, inplace=False) ]
        # self.model = nn.Sequential(*model)

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        points = self.soft_argmax(d1[:,:,:,:])
        points = points.view(points.shape[0], -1)
        # print points.shape
        # points = self.model(points)
        points = self.linear1(points)
        points = self.LeakyReLU(points)
        points = self.linear2(points)
        points = self.LeakyReLU(points)
        points = points.view(points.shape[0], 25, -1)
        return d1, points
