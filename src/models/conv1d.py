import PIL
import time, json, collections, math
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from einops import rearrange, repeat

class Conv1d(torch.nn.Module):
    def __init__(self, params):
        super(Conv1d, self).__init__()
        self.name = "conv1"
        self.params = params
        net_params = params['net']
        data_params = params['data']

        self.spectral_size = data_params.get("spectral_size", 200)
        self.num_classes = data_params.get("num_classes", 16)
        self.patch_size = data_params.get("patch_size", 1)

        self.out_channels=128

        self.layer1 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv1d(1, out_channels=self.out_channels, kernel_size=16, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm1d(self.out_channels)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool1d(kernel_size=2, stride=2))
        ]))

        self.flatten = nn.Flatten()
        self.mlp_head = nn.LazyLinear(self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0]  * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batch, channel, spe, height, width = x.shape
        assert height == 1 and width == 1
        x = torch.reshape(x, [batch, channel, spe])
        h = self.layer1(x)
        h = self.flatten(h)
        h = self.mlp_head(h)
        return h

        
if __name__ == '__main__':
    path_param = './params/cross_param.json'
    with open(path_param, 'r') as fin:
        param = json.loads(fin.read())
    model = Conv1d(param)
    model.eval()
    print(model)
    input = torch.randn(3, 1, 200, 1, 1)
    y = model(input)
    print(y.shape)
