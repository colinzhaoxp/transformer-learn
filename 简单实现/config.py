import torch
import torch.nn as nn
import numpy as np
import math


class Config(object):
    '''
    用来提前配置好参数
    '''
    def __init__(self):
        # 这个是词典所拥有的单词的个数
        self.vocab_size = 6
        # 这个是单词的编码的长度
        self.d_model = 20
        self.n_heads = 2

        # 需要保证单词的维度可以被多头注意力均分
        assert self.d_model % self.n_heads == 0
        dim_k = d_model // n_heads
        dim_v = d_model // n_heads
        
        # 这个是为了限制每个输入语句的最长长度使用
        self.padding_size = 30
        # 
        self.UNK = 5
        self.PAD = 4

        self.N = 6
        self.p = 0.1