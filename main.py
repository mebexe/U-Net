import torch
import troch.nn as nn
import torchvision.transformers.functional as VF
import numpy as np

class BLOCK(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(BLOCK,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3,Stride=1,padding=0), #,bias=False & padding = 1
            # nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3,Stride=1,padding=0), #,bias=False & padding = 1
            # nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(outchannel)
            )


    def forward(self, x):
        return self.block(x)

    class UNET(nn.Module):
        def __init__(self, img_channel, num_classes, feature_size=[64, 128, 256, 512]):
            super(UNET, self).__init__()
            self.downLayers = nn.ModuleList()
            self.upLayers = nn.ModuleList()
            self.bottleneck = BLOCK(feature_size[-1], feature_size[-1]*2)
            self.last_layer = nn.Conv2d(feature_size[0], num_classes,kernel_size=1, stride=1, padding=0)
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

            for f in feature_size:
                self.downlayer.append(BLOCK(img_channel, f))
                img_channel = f

            for f in reversed(feature_size):
            self.upLayers.append(nn.ConvTranspose2d(f*2, f, kernel_size=2 , stride=2))
            self.upLayers.append(BLOCK(f*2, f))





