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

        #  print (a.shape)
        #  print (b.shape)

        print('  ')
        print('  - Initializing...')
        print('  - Reading Pre-trained Knowledge Embeddings...')

        demo_pre_trained = np.load('./kgraphs/pre-trained/liberal.RatatE.128/entity_embedding.npy')
        rep_pre_trained = np.load('./kgraphs/pre-trained/conservative.RotatE.128/entity_embedding.npy')

        common_knowledge = []
        rep_knowledge = []
        demo_knowledge = []
        rep = 0
        demo = 0

        for idx in range(vocab_size):
            mapping = 0
            for j, vocab_idx in enumerate(knowledge_indices['rep']):
                if idx != 0 and idx == vocab_idx:
                    common_knowledge.append(np.zeros(embed_size))
                    mapping = 1
                    break
            if mapping == 0:
                common_knowledge.append(np.zeros(embed_size))

            mapping = 0
            for j, vocab_idx in enumerate(knowledge_indices['rep']):
                if idx != 0 and idx == vocab_idx:
                    rep_knowledge.append(rep_pre_trained[j])
                    #  rep_knowledge.append(np.zeros(embed_size))
                    mapping = 1
                    rep += 1
                    break
            if mapping == 0:
                rep_knowledge.append(np.zeros(embed_size))

            mapping = 0
            for j, vocab_idx in enumerate(knowledge_indices['demo']):
                if idx != 0 and idx == vocab_idx:
                    demo_knowledge.append(demo_pre_trained[j])
                    #  demo_knowledge.append(np.zeros(embed_size))
                    mapping = 1
                    demo += 1
                    break
            if mapping == 0:
                demo_knowledge.append(np.zeros(embed_size))

        print (rep)
        print (demo)

        self.common_knowledge = nn.Embedding.from_pretrained(torch.FloatTensor(common_knowledge))
        self.demo_knowledge = nn.Embedding.from_pretrained(torch.FloatTensor(rep_knowledge))
        self.rep_knowledge = nn.Embedding.from_pretrained(torch.FloatTensor(demo_knowledge))

        self.fuse_knowledge_fc = nn.Linear(embed_size*2, embed_size)

        encoder_layers = TransformerEncoderLayer(embed_size, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.fc = nn.Linear(embed_size, num_class)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.fuse_knowledge_fc.weight.data.uniform_(-initrange, initrange)
        self.fuse_knowledge_fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, texts: Tensor) -> Tensor:
        """
        Args:
            texts: Tensor, shape [batch_size, seq_len]

        Returns:
            output: Tensor, shape[]
        """

        # word embeddings with position encoding
        word_embeddings = self.embeddings(texts) * math.sqrt(self.embed_size)
        emb_with_pos = self.pos_encoder(word_embeddings)
        #  print (emb_with_pos.size())

        emb_with_ckwldg = emb_with_pos + self.common_knowledge(texts)
        #  print (emb_with_ckwldg.size())

        demo_knwldg = emb_with_ckwldg + self.demo_knowledge(texts)
        rep_knwldg = emb_with_ckwldg + self.rep_knowledge(texts)

        # concate and pass a FC layer
        emb_with_knowledge = self.fuse_knowledge_fc(torch.cat((demo_knwldg, rep_knwldg), 2))


        # word-level self-attention layers
        word_embeddings = self.transformer_encoder(emb_with_knowledge)
        #  word_embeddings = self.transformer_encoder(emb_with_ckwldg)
        #  print (texts.size()) # b * seq_len
        #  print (word_embeddings.size()) # b * seq_len * d_model

        #  for i, text in enumerate(texts): # # of batches
            #  s_count = 0
            #  sentence = None
            #  doc = []
            #  num_sentences = 0
            #  for j, word_idx in enumerate(text): # document length (# of words) # 30?
                #  if word_idx == 1 and sentence is not None: # if sentence seperator
                    #  sentence /= s_count
                    #  doc.append(sentence)

                    #  sentence = None
                    #  s_count = 0
                    #  num_sentences += 1
                #  else:
                    #  if s_count == 0:
                        #  sentence = word_embeddings[i][j] 
                        #  s_count += 1
                    #  else:
                        #  sentence += word_embeddings[i][j] 
                        #  s_count += 1

                #  if num_sentences > 30:
                    #  break
            #  if num_sentences < 30:
                #  for _ in range(30 - num_sentences):
                    #  doc.append(self.embeddings(torch.LongTensor([[0]])))

            #  print (torch.stack(doc).size()) # s * d_model
        #  print (sentence.size()) # b * s * d_model

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

        x = x + self.common_knowledge(x)

        x = x + self.demo_knowledge(x)
        x = x + self.rep_knowledge(x)

        output = self.fc(word_embeddings)
        return output


