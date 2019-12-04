import torch
import torch.nn as nn

x = torch.ones(1, 1,10,4)
print(x)

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 4))
print( "conv.weight", conv.weight.size() )
print( "conv.bias", conv.bias.size() )
conv.weight.data = torch.ones((1,1,1,4))
conv.bias.data = torch.zeros(1)

y = conv(x)

print(y)
print(y.size())

