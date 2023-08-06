import torch
from ae_utils_exp import s_init


class enc_CIFAR10(torch.nn.Module):
    def __init__(self, lat=1):
        super(enc_CIFAR10, self).__init__()
        use_bias = True
        self.w0e = s_init(torch.nn.Linear(32, 200, bias=use_bias))
        self.w1e = s_init(torch.nn.Linear(200, 80, bias=use_bias))
        self.w2e = s_init(torch.nn.Linear(80, 40, bias=use_bias))
        self.w3e = s_init(torch.nn.Linear(40, lat, bias=use_bias))
        self.act = torch.nn.SELU(True)

    def forward(self, x):
        x = self.act(self.w0e(x))
        x = self.act(self.w1e(x))
        x = self.act(self.w2e(x))
        return self.w3e(x)

class dec_CIFAR10(torch.nn.Module):
    def __init__(self, lat=1):
        super(dec_CIFAR10, self).__init__()
        use_bias = True
        self.w3d = s_init(torch.nn.Linear(lat, 40, bias=use_bias))
        self.w2d = s_init(torch.nn.Linear(40, 80, bias=use_bias))
        self.w1d = s_init(torch.nn.Linear(80, 200, bias=use_bias))
        self.w0d = s_init(torch.nn.Linear(200, 32, bias=use_bias))
        self.act = torch.nn.SELU(True)

    def forward(self, x):
        x = self.act(self.w3d(x))
        x = self.act(self.w2d(x))
        x = self.act(self.w1d(x))
        return self.w0d(x)