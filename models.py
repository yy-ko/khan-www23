import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class KhanModel(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, num_class: int):
        super(KhanModel, self).__init__()
        #  self.encoder = nn.Embedding(ntoken, d_model)
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True) 
        # (TODO) define our model here
        #  self.d_model = 
        #  self.pos_encoder = PositionalEncoding(d_model, dropout)
        #  encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        #  self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text: Tensor, offsets: Tensor) -> Tensor:
    #  def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            text: Tensor, shape [?, batch_size]
            offsets: Tensor, shape [?, ?]

        Returns:
            output: Tensor of shape [seq_len, batch_size, ntoken]
        """
        word_embeddings = self.embedding(text, offsets)
        output = self.fc(word_embeddings)

        #  word_embeddings = self.pos_encoder(src)
        #  output = self.transformer_encoder(src, src_mask)
        #  output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


