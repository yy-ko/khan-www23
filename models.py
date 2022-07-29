import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class KHANModel(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int, nhead: int, d_hid: int, nlayers: int, dropout: float, num_class: int):
        super(KHANModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embed_size = embed_size

        #  self.knowledge_encoder = KnowledgeEncoding()
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.fc = nn.Linear(embed_size, num_class)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    #  def forward(self, texts: Tensor, offsets: Tensor) -> Tensor:
    def forward(self, texts: Tensor) -> Tensor:
        """
        Args:
            texts: Tensor, shape [batch_size, seq_len]

        Returns:
            output: Tensor, shape[]
        """

        # position and knowledge encoding in word-level embeddings
        word_embeddings = self.embeddings(texts) * math.sqrt(self.embed_size)
        emb_with_pos = self.pos_encoder(word_embeddings)
        emb_with_cknwlg = self.cknowledge_encoder(emb_with_pos)

        # (TODO) domain-specific knowledge encoding
        #  emb_with_dem = self.democratic_knowledge_encoder(emb_with_cknwlg)
        #  emb_with_rep = self.republican_cknowledge_encoder(emb_with_cknwlg)
        # concate and pass a FC layer

        # word-level self-attention layers
        word_embeddings = self.transformer_encoder(word_embeddings)

        # (TODO) setentence-level self-attention layers + title-attention
        word_embeddings = word_embeddings.mean(dim=1)

        output = self.fc(word_embeddings)
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


class KnowledgeEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
