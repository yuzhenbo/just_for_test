import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable

import collections
from itertools import repeat

OPS = {
    'conv1':lambda shape_in,shape_out:conv(shape_in,shape_out,1,False),
    'conv3':lambda shape_in,shape_out:conv(shape_in,shape_out,3,False),
    'conv5':lambda shape_in,shape_out:conv(shape_in,shape_out,5,False),
    'conv7':lambda shape_in,shape_out:conv(shape_in,shape_out,7,False),
    'conv9':lambda shape_in,shape_out:conv(shape_in,shape_out,9,False),
    'conv1b':lambda shape_in,shape_out:conv(shape_in,shape_out,1,True),
    'conv3b':lambda shape_in,shape_out:conv(shape_in,shape_out,3,True),
    'conv5b':lambda shape_in,shape_out:conv(shape_in,shape_out,5,True),
}

class conv(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 kernel,
                 use_bias):
        super(conv, self).__init__()
        pad=kernel//2
        self.conv=nn.Conv2d(input_shape[0],
                            output_shape[0],
                            kernel,
                            (input_shape[1]//output_shape[1],input_shape[2]//output_shape[2]),
                            pad,
                            bias=use_bias)
    def forward(self,x):
        return self.conv(x)