import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F


class LuongConcatAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim):
        assert dec_hidden_dim == enc_hidden_dim, 'This attention requires same dimensionality'
        super().__init__()
        self.W = nn.Linear(2*enc_hidden_dim, attn_dim, bias = False)
        self.v = nn.Linear(attn_dim, 1, bias = False)

    def forward(self, encoder_outputs, decoder_hidden):
        # decoder -> (B, D)
        # encoder -> (B, T, D)
        B, T, D = encoder_outputs.shape

        # Expand decoder along time dimension to concat with encoder
        decoder_hidden = decoder_hidden.unsqueeze(1).expand(-1, T, -1) # (B, D) -> (B, T, D)

        # Concat along last dim
        concat = torch.cat((decoder_hidden, encoder_outputs), dim = 2) # (B, T, 2*D)

        # Apply non-linear transformations
        energy = torch.tanh(self.W(concat)) # (B, T, A)
        scores = self.v(energy).squeeze(2) # (B, T, 1) -> (B, T)

        # Attention weights
        attention_weights = F.softmax(scores, dim = 1)#) #(B, T)

        # Context vector as weighted sum
        context_vector = torch.sum(attention_weights.unsqueeze(2)*encoder_outputs, dim = 1) # (B, D)

        return context_vector, attention_weights