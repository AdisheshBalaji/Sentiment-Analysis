import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F

# Eg use
# attention = BahdanauAttention(enc_hidden_dim, dec_hidden_dim, attn_dim)
# context_vector, attention_weights = attention(encoder_outputs, decoder_hidden)

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim):
        super().__init__()
        # Linear projection for encoder
        self.W_enc = nn.Linear(enc_hidden_dim, attn_dim, bias = False)
        # Linear projection for decoder hidden state
        self.W_dec = nn.Linear(dec_hidden_dim, attn_dim, bias = False)
        # Scalar projection score similarity 
        self.v = nn.Linear(attn_dim, 1, bias = False)


    # Forward Propogation Method
    def forward(self, encoder_outputs, decoder_hidden):

        # Encoder outputs size: (B, T, H_enc)
        # Decoder outputs size: (B, H_dec)


        # Projecting the encoder hidden states(h_is)
        Wh = self.W_enc(encoder_outputs) # (B, T, attn_dim)
        # Projecting the decoder final hidden state(s_t)
        Us = self.W_dec(decoder_hidden).unsqueeze(1) # (B, 1, attn_dim)
        # Linear Combination with softmax 
        scores = self.v(F.tanh(Wh + Us)).squeeze(2) # (B, T)
        # Obtaining attention weights (alpha_t)
        attention_weights = F.softmax(scores, dim = 1) # (B, T)
        attention_weights = attention_weights.unsqueeze(1) # (B, 1, T)
        # Obtaining context vector (c_t)
        # print("attention_weights shape before bmm", attention_weights.shape)
        # print("encoder_outputs shape before bmm", encoder_outputs.shape)

        context_vector = torch.bmm(attention_weights, encoder_outputs) # (B, 1, H_enc)
        context_vector = context_vector.squeeze(1) # (B, H_enc)

        return context_vector, attention_weights