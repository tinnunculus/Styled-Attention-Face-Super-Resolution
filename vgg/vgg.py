from collections import namedtuple
from torchvision import models
import torch


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained = True).features                                              # vgg의 layer module들을 나타내 주는 것.
        self.slice1 = torch.nn.Sequential()                                                                             # Modules will be added to it in the order they are passed in the constructor.
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):                                                                                              # vgg의 1,2,3,4 Layer의 module들을 쌓아 올린다.
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():                                                                             # parameters 가 module의 method인가보다. 상속 받았기 때문에 사용할 수 있다.
                param.requires_grad = False                                                                             # requires grad는 default가 True이다.

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])                            # VggOutputs 라는 tuple subclass를 만든다.
        #vgg_outputs.relu1_2 or vgg_outputs[0]
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out