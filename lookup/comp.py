from lookup import Lookup_table as lt
from ops_comp import OPS
import os
import sys
import torch

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='3'
    torch.backends.cudnn.benchmark = True
    shapes=[
        (3,64,64),
        (64,32,32),
        (64,32,32),
        (128,16,16),
        (128,16,16),
        (128,16,16),
        (256,8,8),
        (256,8,8),
    ]
    l=lt(OPS,shapes,name='comp')
    ss=l.get()
    for s in ss:
        print(s)