import torch.nn as nn
from torchsummary import summary


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def summary(self, input_size, device):
        print(summary(self.to(device), input_size=input_size))
