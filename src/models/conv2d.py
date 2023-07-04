import PIL
import time, json, collections, math
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from einops import rearrange, repeat

class Conv2d(torch.nn.Module):
    def __init__(self, params):
        super(Conv2d, self).__init__()
        self.name = "conv4"
        self.params = params
        net_params = params['net']
        data_params = params['data']

        self.spectral_size = data_params.get("spectral_size", 200)
        self.num_classes = data_params.get("num_classes", 16)
        self.patch_size = data_params.get("patch_size", 13)

        self.layer1 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(self.spectral_size, 8, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(8)),
          ('relu',    nn.ReLU()),
        #   ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer2 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(16)),
          ('relu',    nn.ReLU()),
        #   ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer3 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(32)),
          ('relu',    nn.ReLU()),
        #   ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        # self.layer4 = nn.Sequential(collections.OrderedDict([
        #   ('conv',    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)),
        #   ('bn',      nn.BatchNorm2d(512)),
        #   ('relu',    nn.ReLU()),
        #   #('avgpool', nn.AvgPool2d(kernel_size=4))
        #   ('glbpool', nn.AdaptiveAvgPool2d(1))
        # ]))

        self.flatten = nn.Flatten()
        self.mlp_head = nn.LazyLinear(self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.flatten(h)
        h = self.mlp_head(h)
        return h

        
if __name__ == '__main__':
    path_param = './params/cross_param.json'
    with open(path_param, 'r') as fin:
        param = json.loads(fin.read())
    model = Conv2d(param)
    model.eval()
    print(model)
    input = torch.randn(3, 200, 9, 9)
    y = model(input)
    print(y.shape)
