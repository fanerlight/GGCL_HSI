import PIL
import time, json, collections, math
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from einops import rearrange, repeat

class Conv3d(torch.nn.Module):
    def __init__(self, params):
        super(Conv3d, self).__init__()
        self.name = "conv3d"
        self.params = params
        net_params = params['net']
        data_params = params['data']

        self.spectral_size = data_params.get("spectral_size", 200)
        self.num_classes = data_params.get("num_classes", 16)
        self.patch_size = data_params.get("patch_size", 13)

        a,b,c,d = 1,8,16,32
        self.layer1 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv3d(a, b, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
          ('bn',      nn.BatchNorm3d(b)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.Conv3d(b, b, (3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))),
        ]))

        self.layer2 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv3d(b, c, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
          ('bn',      nn.BatchNorm3d(c)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.Conv3d(c, c, (3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))),
        ]))

        self.layer3 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv3d(c, d, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
          ('bn',      nn.BatchNorm3d(d)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.Conv3d(d, d, (3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))),
        ]))

        self.flatten = nn.Flatten()
        self.mlp_head = nn.LazyLinear(self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
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
    model = Conv3d(param)
    model.eval()
    print(model)
    input = torch.randn(3, 1, 200, 9, 9)
    y = model(input)
    print(y.shape)