from curses import def_prog_mode
from tkinter import N
from turtle import forward, pos
from urllib.parse import non_hierarchical
import torch
import torch.nn as nn
from torch.nn import functional as F

import math


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, p)



class Positional_Encoding:
    def __init__(self, max_len, d_model):
        self.max_len = max_len
        self.d_model = d_model
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, atten_mask=None):
        """
        Q: [batch_size, n_heads, len_q, d_q]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v, k_v]
        len_q == len_v, d_q == d_v
        """
        d = queries.shape[-1]
        # scores: [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(queries, keys.transpose(-1, -2) / math.sqrt(d))
        scores.masked_fill_(atten_mask, -1e9)

        attention = F.softmax(scores, dim=-1)

        # context: [batch_size, n_heads, len_q, k_v]
        context = torch.matmul(attention, values)
        return context, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_hiddens, n_heads):
        super().__init__()
        self.d_hiddens = d_hiddens
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_hiddens * n_heads)
        self.W_k = nn.Linear(d_model, d_hiddens * n_heads)
        self.W_v = nn.Linear(d_model, d_hiddens * n_heads)
        self.linear = nn.Linear(d_hiddens * n_heads, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, atten_mask):
        # Q: [batch, len_q, d_model]
        residual, batch_size = Q, Q.shape[0]
        q_s = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_hiddens).transpose(1, 2) # batch*n_heads*len_q*d_hiddens
        k_s = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_hiddens).transpose(1, 2)
        v_s = self.w_v(V).view(batch_size, -1, self.n_heads, self.d_hiddens).transpose(1, 2)

        atten_mask = atten_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, k_v]
        # atten: [batch_size, n_heads, len_q, len_k]
        context, atten = ScaledDotProductAttention(q_s, k_s, v_s, atten_mask)

        # context: [batch_size, len_q, n_heads * k_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_hiddens)
        output = self.linear(context)

        return self.layer_norm(output + residual), atten


class PoswiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, inputs):
        # inputs: [batch_size, len_q, d_model]
        residual = inputs
        outputs = F.relu(self.conv1(inputs.transpose(1, 2)))
        outputs = self.conv2(outputs).transpose(1, 2)
        return self.layer_norm(outputs + residual)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_hiddens, n_heads, d_ff):
        super().__init__()
        self.enc_self_atten = MultiHeadAttention(d_model, d_hiddens, n_heads)
        self.pos_ffn = PoswiseFeedForward(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForward()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

