import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms 
from torchvision.transforms import ToTensor
from initializenn import *

NET_TYPES = {'vgg16' : torchvision.models.vgg16}
                
class VGG16(nn.Module):
    def __init__(self,
                 ipt_size=(512, 512), 
                 pretrained=True, 
                 net_type='vgg16', 
                 num_classes=2,
                 init='kaimingNormal',
                 train=True):
        super(VGG16, self).__init__()

        #Mode initialization
        self.tr = train

        # load convolutional part of vgg
        assert net_type in NET_TYPES, "Unknown vgg_type '{}'".format(net_type)
        net_loader = NET_TYPES[net_type]
        net = net_loader(pretrained=pretrained)
        self.features = net.features

        # init fully connected part of vgg
        test_ipt = Variable(torch.zeros(1,3,ipt_size[0],ipt_size[1]))
        test_out = net.features(test_ipt)
        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x) 
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.tr:
            return F.log_softmax(x) #Use this for training
        else:
            return x #Use this to get the probability scores, and apply softmax on the output scores
