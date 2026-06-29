import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F


# Encoder outputs size: (B, T, H_enc)
# Decoder outputs size: (B, H_dec)

class LuongDotAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim):
        assert dec_hidden_dim == enc_hidden_dim, 'This attention requires same dimensionality'
        super().__init__()
    

    def forward(self, encoder_outputs, decoder_hidden):

        
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2))  # (B, T, H) x (B, H, 1) -> (B, T, 1)
        attention_weights = F.softmax(scores, dim = 1) # (B, T, 1)

        # (B, T, 1) * (B, T, H) 
        # Reshape the attn weights into (1, T) and then do btt
        context_vector = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs) # (B, 1, H)
        attention_weights = attention_weights.squeeze(2) #(B, T, 1)->(B, T)

        return context_vector, attention_weights