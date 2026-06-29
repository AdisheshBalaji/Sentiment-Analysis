import torch
import torch.nn as nn

from attention.bahdanau import BahdanauAttention
from attention.luonggeneral import LuongGeneralAttention
from attention.luongconcat import LuongConcatAttention
from attention.luongdot import LuongDotAttention

# ----------------------------
# Text Utilities
# ----------------------------

def encode_text(text, word2idx):
    return [word2idx.get(word, word2idx["<UNK>"]) for word in text.split()]


def pad_sequence_to_length(seq, max_len, pad_idx):
    return seq + [pad_idx] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]


# ----------------------------
# Model Definitions
# ----------------------------

class VanillaLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size,
                 attention_class=None, attn_dim=None, dropout=0.3):
        super().__init__()

        self.hidden_size = hidden_size
        num_embeddings, embed_size = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)

        self.use_attention = attention_class is not None
        if self.use_attention:
            self.attention = attention_class(hidden_size, hidden_size, attn_dim)
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = x.device
        x = self.embedding(x)

        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.dropout_layer(out)

        if self.use_attention:
            context, attn_weights = self.attention(out, hn[-1])
            context = context.squeeze(1)
            combined = torch.cat((context, hn[-1]), dim=1)
            output = self.fc(combined)
            return output, attn_weights.squeeze(1)
        else:
            output = self.fc(hn[-1])
            return output


class VanillaRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size,
                 attention_class=None, attn_dim=None, dropout=0.3):
        super().__init__()

        self.hidden_size = hidden_size
        num_embeddings, embed_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)

        self.use_attention = attention_class is not None
        if self.use_attention:
            self.attention = attention_class(hidden_size, hidden_size, attn_dim)
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = x.device
        x = self.embedding(x)

        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, hn = self.rnn(x, h0)
        out = self.dropout_layer(out)

        if self.use_attention:
            context, attn_weights = self.attention(out, hn[-1])
            context = context.squeeze(1)
            combined = torch.cat((context, hn[-1]), dim=1)
            output = self.fc(combined)
            return output, attn_weights.squeeze(1)
        else:
            output = self.fc(out[:, -1, :])
            return output


class BidirectionalRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size,
                 attention_class=None, attn_dim=None, dropout=0.3):
        super().__init__()

        self.hidden_size = hidden_size
        num_embeddings, embed_size = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.birnn = nn.RNN(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout_layer = nn.Dropout(dropout)

        self.use_attention = attention_class is not None
        if self.use_attention:
            self.attention = attention_class(2 * hidden_size, 2 * hidden_size, attn_dim)
            self.fc = nn.Linear(4 * hidden_size, output_size)
        else:
            self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        device = x.device
        x = self.embedding(x)

        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(device)
        out, hn = self.birnn(x, h0)
        out = self.dropout_layer(out)

        final_hidden = torch.cat((hn[0], hn[1]), dim=1)

        if self.use_attention:
            context, attn_weights = self.attention(out, final_hidden)
            context = context.squeeze(1)
            combined = torch.cat((context, final_hidden), dim=1)
            output = self.fc(combined)
            return output, attn_weights.squeeze(1)
        else:
            output = self.fc(final_hidden)
            return output


class BidirectionalLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size,
                 attention_class=None, attn_dim=None, dropout=0.3):
        super().__init__()

        self.hidden_size = hidden_size
        num_embeddings, embed_size = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.bilstm = nn.LSTM(embed_size, hidden_size, batch_first=True,
                              bidirectional=True, num_layers=1)
        self.dropout_layer = nn.Dropout(dropout)

        self.use_attention = attention_class is not None
        if self.use_attention:
            self.attention = attention_class(2 * hidden_size, 2 * hidden_size, attn_dim)
            self.fc = nn.Linear(4 * hidden_size, output_size)
        else:
            self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        device = x.device
        x = self.embedding(x)

        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(device)

        out, (hn, cn) = self.bilstm(x, (h0, c0))
        out = self.dropout_layer(out)

        final_hidden = torch.cat((hn[0], hn[1]), dim=1)

        if self.use_attention:
            context, attn_weights = self.attention(out, final_hidden)
            context = context.squeeze(1)
            combined = torch.cat((context, final_hidden), dim=1)
            output = self.fc(combined)
            return output, attn_weights.squeeze(1)
        else:
            output = self.fc(final_hidden)
            return output


# ----------------------------
# Registries — used by model_loader.py
# ----------------------------

MODEL_REGISTRY = {
    "vanilla_lstm":   VanillaLSTM,
    "vanilla_rnn":    VanillaRNN,
    "bidirectional_rnn":  BidirectionalRNN,
    "bidirectional_lstm": BidirectionalLSTM,
}

ATTENTION_REGISTRY = {
    "bahdanau":      BahdanauAttention,
    "luong_general": LuongGeneralAttention,
    "luong_concat":  LuongConcatAttention,
    "luong_dot":     LuongDotAttention,
    "none":          None,
}