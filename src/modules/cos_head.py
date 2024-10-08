import math
import torch
from torch import nn
from torch.nn import functional as F


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=1):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))

        self.sigma_init_value = abs(sigma)
        self.sigma = nn.Parameter(torch.tensor(abs(sigma)), requires_grad=sigma < 0)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.sigma.data.fill_(self.sigma_init_value)

    def forward(self, input):
        weight = F.normalize(self.weight, p=2, dim=1)
        input = F.normalize(input, p=2, dim=1)
        out = F.linear(input, weight) * self.sigma
        return out
