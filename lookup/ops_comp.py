import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable

import collections
from itertools import repeat

OPS = {
    'conv32':lambda shape_in,shape_out:conv(shape_in,shape_out,32,False),
    'conv64':lambda shape_in,shape_out:conv(shape_in,shape_out,64,False),
    'conv128':lambda shape_in,shape_out:conv(shape_in,shape_out,128,False),
    'conv256':lambda shape_in,shape_out:conv(shape_in,shape_out,256,False),
    'conv512':lambda shape_in,shape_out:conv(shape_in,shape_out,512,False),
    'conv32b':lambda shape_in,shape_out:conv(shape_in,shape_out,32,True),
    'conv64b':lambda shape_in,shape_out:conv(shape_in,shape_out,64,True),
    'conv128b':lambda shape_in,shape_out:conv(shape_in,shape_out,128,True),
}

class conv(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 mid_channels,
                 use_bias):
        super(conv, self).__init__()
        kernel=3
        pad=kernel//2
        self.conv1=nn.Conv2d(input_shape[0],
                            mid_channels,
                             kernel,
                            (input_shape[1]//output_shape[1],input_shape[2]//output_shape[2]),
                            pad,
                            bias=use_bias)
        self.conv2 = nn.Conv2d(mid_channels,
                               output_shape[0],
                               kernel,
                               1,
                               pad,
                               bias=use_bias)
    def forward(self,x):
        return self.conv2(self.conv1(x))