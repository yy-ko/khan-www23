import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class KHANModel(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int, nhead: int, d_hid: int, nlayers: int, dropout: float, num_class: int, knowledge_indices):
        super(KHANModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embed_size = embed_size
        self.pos_encoder = PositionalEncoding(embed_size, dropout)

        print('  ')
        print('  - Initializing...')
        print('  - Reading Pre-trained Knowledge Embeddings...')

        #  common_knowledge_path = './kgraphs/pre-trained/FB15K.RotatE.'
        common_knowledge_path = './kgraphs/pre-trained/YAGO.RotatE.'
        demo_knowledge_path = './kgraphs/pre-trained/liberal.RotatE.'
        rep_knowledge_path = './kgraphs/pre-trained/conservative.RotatE.'

        if embed_size == 128:
            common_knowledge_path += '128/entity_embedding.npy'
            demo_knowledge_path += '128/entity_embedding.npy'
            rep_knowledge_path += '128/entity_embedding.npy'
        elif embed_size == 256:
            common_knowledge_path += '256/entity_embedding.npy'
            demo_knowledge_path += '256/entity_embedding.npy'
            rep_knowledge_path += '256/entity_embedding.npy'
        elif embed_size == 512:
            common_knowledge_path += '512/entity_embedding.npy'
            demo_knowledge_path += '512/entity_embedding.npy'
            rep_knowledge_path += '512/entity_embedding.npy'
        elif embed_size == 1024:
            common_knowledge_path += '1024/entity_embedding.npy'
            demo_knowledge_path += '1024/entity_embedding.npy'
            rep_knowledge_path += '1024/entity_embedding.npy'
        else:
            print ('Wrong embedding dimension! Dimension should be 128, 256, 512, or 1024')
            sys.exit(1)


        #  common_pre_trained = np.load(common_knowledge_path)
        #  demo_pre_trained = np.load(demo_knowledge_path)
        #  rep_pre_trained = np.load(rep_knowledge_path)

        #  common_knowledge = []
        #  rep_knowledge = []
        #  demo_knowledge = []
        #  rep = 0
        #  demo = 0

        #  for idx in range(vocab_size):
            #  mapping = 0
            #  for j, vocab_idx in enumerate(knowledge_indices['common']):
                #  if idx != 0 and idx == vocab_idx:
                    #  common_knowledge.append(common_pre_trained[j])
                    #  #  common_knowledge.append(np.zeros(embed_size))
                    #  mapping = 1
                    #  break
            #  if mapping == 0:
                #  common_knowledge.append(np.zeros(embed_size))

            #  mapping = 0
            #  for j, vocab_idx in enumerate(knowledge_indices['rep']):
                #  if idx != 0 and idx == vocab_idx:
                    #  rep_knowledge.append(rep_pre_trained[j])
                    #  #  rep_knowledge.append(np.zeros(embed_size))
                    #  mapping = 1
                    #  rep += 1
                    #  break
            #  if mapping == 0:
                #  rep_knowledge.append(np.zeros(embed_size))

            #  mapping = 0

            #  for j, vocab_idx in enumerate(knowledge_indices['demo']):
                #  if idx != 0 and idx == vocab_idx:
                    #  demo_knowledge.append(demo_pre_trained[j])
                    #  #  demo_knowledge.append(np.zeros(embed_size))
                    #  mapping = 1
                    #  demo += 1
                    #  break
            #  if mapping == 0:
                #  demo_knowledge.append(np.zeros(embed_size))


        #  self.common_knowledge = nn.Embedding.from_pretrained(torch.FloatTensor(common_knowledge))
        #  self.demo_knowledge = nn.Embedding.from_pretrained(torch.FloatTensor(rep_knowledge))
        #  self.rep_knowledge = nn.Embedding.from_pretrained(torch.FloatTensor(demo_knowledge))

        self.fuse_knowledge_fc = nn.Linear(embed_size*2, embed_size)

        encoder_layers = TransformerEncoderLayer(embed_size, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        sentence_encoder_layers = TransformerEncoderLayer(embed_size, nhead, d_hid, dropout)
        self.sentence_transformer = TransformerEncoder(encoder_layers, nlayers)

        self.fc = nn.Linear(embed_size, num_class)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.fuse_knowledge_fc.weight.data.uniform_(-initrange, initrange)
        self.fuse_knowledge_fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    #  def forward(self, texts: Tensor) -> Tensor:
    def forward(self, sentences: Tensor) -> Tensor:
        """
        Args:
            texts: Tensor, shape [batch_size, seq_len]
        Returns:
            output: Tensor, shape[]
        """

        # word embeddings with position encoding
        #  texts = torch.flatten(sentences, start_dim=1)

        sentence_embeddings = []
        for texts in sentences:
            word_embeddings = self.embeddings(texts) * math.sqrt(self.embed_size)
            emb_with_pos = self.pos_encoder(word_embeddings)

        #  emb_with_ckwldg = emb_with_pos + self.common_knowledge(texts)
        #  demo_knwldg = emb_with_ckwldg + self.demo_knowledge(texts)
        #  rep_knwldg = emb_with_ckwldg + self.rep_knowledge(texts)

        #  concate and pass a FC layer
        #  emb_with_knowledge = self.fuse_knowledge_fc(torch.cat((demo_knwldg, rep_knwldg), 2))


        # word-level self-attention layers
        #  word_embeddings = self.transformer_encoder(emb_with_knowledge)
            word_embeddings = self.transformer_encoder(emb_with_pos)

        # (TODO) setentence-level self-attention layers + title-attention
        #  word_embeddings = word_embeddings.mean(dim=1)
            sentence_embedding = word_embeddings.mean(dim=1)
            sentence_embeddings.append(sentence_embedding)

        sentence_embeddings = torch.stack(sentence_embeddings)
        sentence_embeddings = self.sentence_transformer(sentence_embeddings)

        doc_embeddings = sentence_embeddings.mean(dim=1)
        
        output = self.fc(doc_embeddings)
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

    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        #  self.dropout = nn.Dropout(p=dropout)


        self.fc = nn.Linear(embed_size*2, embed_size)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        #  x = x + self.pe[:x.size(0)]

        k = x + self.common_knowledge(x)

        x = x + self.demo_knowledge(x)
        x = x + self.rep_knowledge(x)

        output = self.fc(word_embeddings)
        return output
