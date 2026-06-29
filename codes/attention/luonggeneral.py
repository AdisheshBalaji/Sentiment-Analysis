import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F


class LuongGeneralAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim):
        assert dec_hidden_dim == enc_hidden_dim, 'This attention requires same dimensionality'
        super().__init__()
        self.W = nn.Linear(enc_hidden_dim, enc_hidden_dim, bias = False) # W(d x d)

    def forward(self, encoder_outputs, decoder_hidden):
        Wh = self.W(encoder_outputs) # (B, T, D)
        decoder_hidden = decoder_hidden.unsqueeze(2) # (B, D) -> # (B, D, 1)
        scores = torch.bmm(Wh, decoder_hidden) # (B, T, 1)
        scores = scores.squeeze(2) # (B, T)
        attention_weights = F.softmax(scores, dim = 1) # (B, T)
        context_vector = torch.sum(attention_weights.unsqueeze(2)*encoder_outputs, dim = 1) # (B, D)

        return context_vector, attention_weights
    