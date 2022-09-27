import torch
import torch.nn as nn
import numpy as np
import math
from .config import Config

config = Config()

class Embedding(nn.Moudule):
    '''
    transformer的原始输入是一个文本编码输入
    batch * seq_len, [[1, 3, 0, 4], [2, 1, 3]]
    batch表示样本的个数,也就是有几句话。
    seq_len表示每段话的长度，需要注意的是每段话的长度可能不一样长
    如上面的样本示例中，[1, 3, 0, 4]这段话有4个单词。
    其中的1，表示在当前的词典中，标号为1的单词，也就是说现在单词的表示就是一个数字
    需要通过本模块，将单词用一个向量来表示，而不是单独一个数字。
    '''
    def __init__(self, vocab_size):
        super(Embedding, self).__init__()
        # nn.Embedding类的作用就是创建一个字典向量，vocab_size * config.d_model
        # vocab_size 表示单词的个数
        # config.d_model 表示每个单词使用几个维度进行表示
        self.embedding = nn.Embedding(vocab_size, config.d_model,
                            padding_idx=config.PAD)
        
        def forward(self, x):
            # 需要对每个句子进行处理
            for i in range(len(x)):
                # 如果当前句子的长度小于我们设定的长度，那么就扩充
                # 用特定的符号进行填充即可
                if len(x[i]) < config.padding_size:
                    x[i].extend([config.UNK] * (config.padding_size - len(x[i])))
                else:
                    # 如果过长，那么就将多余的部分去除掉。
                    x[i] = x[i][:config.padding_size]
            # (batch * seq_len)  vocab_size * config.d_model
            # 得到 batch_size * seq_len * d_model
            x = self.embedding(torch.tensor(x))
            return x


class Positional_Encoding(nn.Module):
    '''
    位置编码
    获得单词的embedding之后，还需要对每个位置添加上位置编码信息
    '''
    def __init__(self, d_model):
        super(Positional_Encoding, self).__init__()
        # 每个单词的维度
        self.d_model = d_model

    def forward(self, seq_len, embedding_dim):
        # 位置编码的维度，和输入的维度是相同的。
        # 如一句话的embedding是 seq_len * embedding_dim
        positional_encoding = np.zeros((seq_len, embedding_dim))
        # 对一句话中的每个单词
        for pos in range(positional_encoding.shape[0]):
            # 对单词中的每个向量的值
            for i in range(positional_encoding.shape[1]):
                positional_encoding[pos][i] = math.sin(pos/(1000**(2*i/self.d_model))) \
                    if i %2 == 0 \
                    else math.cos(pos/(10000**(2*i/self.d_model)))
        return torch.from_numpy(positional_encoding)


class Mutihead_Attention(nn.Module):
    '''
    多头注意力
    '''
    def __init__(self, d_model, dim_k, dim_v, n_heads):
        super(Mutihead_Attention, self).__init__()
        assert dim_k % n_heads == 0 and dim_v % n_heads == 0
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads

        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_k)

        self.o = nn.Linear(dim_v, d_model)
        self.norm_fact = 1 / math.sqrt(d_model)

    def generate_mask(self, dim):
        matrix = np.ones((dim, dim))
        mask = torch.Tensor(np.tril(matrix))
        return mask == 1
    
    def forward(self, x, y, requires_mask=False):
        # x: batch * seq_len * d_model
        # Q: n_heads * batch * seq_len * (dim_K // n_heads)
        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        K = self.k(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        V = self.v(y).reshape(-1, x.shape[0], x.shape[1], self.dim_v // self.n_heads)

        # attention_scors: n_heads * batch * seq_len * seq_len
        attention_score = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact
        
        if requires_mask:
            mask = self.generate_mask(x.shape[1])
            attention_score.masked_fill(mask, value=float('-inf'))
        
        # 经过下面的操作，V先回到n_heads * batch * seq_len * (dim_v // n_heads)
        # 然后再回到 batch * seq_len * dim_v
        output = torch.matmul(attention_score, V).reshape(y.shape[0], y.shape[1], -1)
        # output: batch * seq_len * d_model
        output = self.o(output)

        return output

class Feed_Forward(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048):
        super(Feed_Forward, self).__init__()
        self.L1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.L2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        output = self.relu(self.L1(x))
        output = self.L2(output)

        return output

class Add_Norm(nn.Module):
    def __init__(self):
        super(Add_Norm, self).__init__()
        self.dropout = nn.Dropout(config.p)
    
    def forward(self, x, sub_layer, **kwargs):
        sub_output = sub_layer(x, **kwargs)
        x = self.dropout(x + sub_output)
        layer_norm = nn.LayerNorm(x.shape[1:])
        out = layer_norm(x)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.positional_encoding = Positional_Encoding(config.d_model)
        self.muti_atten = Mutihead_Attention(config.d_model, config.dim_k, config.dim_v, 
                        config.n_heads)
        self.feed_forward = Feed_Forward(config.d_model)
        self.add_norm = Add_Norm()
    
    def forward(self, x):
        # x: batch * seq_len, x的类型是普通的list ？ 
        x += self.positional_encoding(x.shape[1], config.d_model)
        output = self.add_norm(x, self.muti_atten, y = x)
        output = self.add_norm(output, self.feed_forward)
        return output
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.positional_encoding = Positional_Encoding(config.d_model)
        self.muti_atten = Mutihead_Attention(config.d_model, config.dim_k, config.dim_v, 
                        config.n_heads)
        self.feed_forward = Feed_Forward(config.d_model)
        self.add_norm = Add_Norm()

    def forward(self, x, encoder_output):
        x += self.positional_encoding(x.shape[1], config.d_model)
        output = self.add_norm(x, self.muti_atten, y=x, requires_mask=True)
        output = self.add_norm(output, self.muti_atten, y=encoder_output, requires_mask=True)
        output = self.add_norm(output, self.feed_forward)

        return output

class Transformer_layer(nn.Module):
    def __init__(self):
        super(Transformer_layer, self).__init__()
        self.encoder = Encoder()
        self.Decoder = Decoder()
    
    def forward(self, x):
        x_input, x_output = x
        encoder_output = self.encoder(x_input)
        decoder_output = self.decoder(x_output, encoder_output)
        return (encoder_output, decoder_output)

class Transformer(nn.Module):
    def __init__(self, N, vocab_size, output_dim):
        super(Transformer, self).__init__()
        self.embedding_input = Embedding(vocab_size = vocab_size)
        self.embedding_output = Embedding(vocab_size = vocab_szie)

        self.output_dim = output_dim
        self.linear = nn.Linear(config.d_model, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.model = nn.Sequential(*[Transformer_layer() for _ in range(N)])

    def forward(self, x):
        x_input, x_output = x
        x_input = self.embedding_input(x_input)
        x_output = self.embedding_input(x_output)

        _, output = self.model((x_input, x_output))

        output = self.linear(output)
        output = self.softmax(output)

        return output




