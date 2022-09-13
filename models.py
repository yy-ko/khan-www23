import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class KHANModel(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int, nhead: int, d_hid: int, nlayers: int, dropout: float, num_class: int, knowledge_indices, alpha, beta):
        super(KHANModel, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embed_size = embed_size

        self.pos_encoder = PositionalEncoding(embed_size, dropout, 2400)
        self.title_pos_encoder = PositionalEncoding(embed_size, dropout, 100)
        self.knowledge_encoder = KnowledgeEncoding(vocab_size, embed_size, knowledge_indices, alpha, beta, dropout) 

        title_encoder_layers = TransformerEncoderLayer(embed_size, nhead, d_hid, dropout, batch_first=True)
        self.title_transformer = TransformerEncoder(title_encoder_layers, nlayers)

        word_encoder_layers = TransformerEncoderLayer(embed_size, nhead, d_hid, dropout, batch_first=True)
        self.word_transformer = TransformerEncoder(word_encoder_layers, nlayers)

        sentence_encoder_layers = TransformerEncoderLayer(embed_size, nhead, d_hid, dropout, batch_first=True)
        self.sentence_transformer = TransformerEncoder(sentence_encoder_layers, nlayers)

        self.title_multihead_attention = nn.MultiheadAttention(embed_size, nhead, dropout, batch_first=True)

        self.classifier = nn.Linear(embed_size, num_class)
        self.init_weights()



    def init_weights(self) -> None:
        initrange = 0.5
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.classifier.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()

    #  def forward(self, texts: Tensor) -> Tensor:
    def forward(self, sentences: Tensor, titles: Tensor) -> Tensor:
        """
        Args:
            sentences: Tensor, shape [batch_size, sentence_len, word_len]
        Returns:
            output: Tensor, shape[batch_size, num_class]
        """
        isHierarchy = True

        if isHierarchy == True:
            title_embeddings = self.embeddings(titles) * math.sqrt(self.embed_size)
            title_embeddings = self.title_pos_encoder(title_embeddings)
            title_embeddings = self.title_transformer(title_embeddings)
            title_embeddings = title_embeddings.mean(dim=1).unsqueeze(1)

            sentence_embeddings = []
            for texts in sentences: # batch_size (# of articles in a batch)
                word_embeddings = self.embeddings(texts) * math.sqrt(self.embed_size)
                residual = word_embeddings
                word_embeddings = self.knowledge_encoder(word_embeddings, texts)
                word_embeddings += residual
                #  word_embeddings = self.layer_norm(word_embeddings + residual)

                word_embeddings = self.pos_encoder(word_embeddings)
                word_embeddings = self.word_transformer(word_embeddings)

                sentence_embedding = word_embeddings.mean(dim=1)
                sentence_embeddings.append(sentence_embedding)

            sentence_embeddings = torch.stack(sentence_embeddings)
            sentence_embeddings = self.pos_encoder(sentence_embeddings)
            sentence_embeddings = self.sentence_transformer(sentence_embeddings)

            # title-attention
            doc_embeddings = sentence_embeddings.mean(dim=1)
            title_embeddings, _ = self.title_multihead_attention(title_embeddings, sentence_embeddings, sentence_embeddings)
            doc_embeddings = title_embeddings.squeeze(1) + doc_embeddings
            #  doc_embeddings = self.layer_norm(title_embeddings.squeeze(1) + doc_embeddings)

            #  output = self.classifier(title_embeddings.squeeze(1))
            output = self.classifier(doc_embeddings)
            return output

        else:
            texts = torch.flatten(sentences, start_dim=1)
            word_embeddings = self.embeddings(texts) * math.sqrt(self.embed_size)
            emb_with_pos = self.pos_encoder(word_embeddings)
            word_embeddings = self.word_transformer(emb_with_pos)
            doc_embeddings = word_embeddings.mean(dim=1)

            output = self.classifier(doc_embeddings)
            return output




class KnowledgeEncoding(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int, knowledge_indices, alpha: float, beta: float, dropout: float = 0.4):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        #  common_knowledge_path = './kgraphs/pre-trained/FB15K.RotatE.'
        common_knowledge_path = './kgraphs/pre-trained/YAGO.RotatE.'
        #  demo_knowledge_path = './kgraphs/pre-trained-revised/liberal.RotatE.'
        #  rep_knowledge_path = './kgraphs/pre-trained-revised/conservative.RotatE.'
        demo_knowledge_path = './kgraphs/pre-trained-revised/liberal.ModE.'
        rep_knowledge_path = './kgraphs/pre-trained-revised/conservative.ModE.'


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

        common_pre_trained = np.load(common_knowledge_path)
        demo_pre_trained = np.load(demo_knowledge_path)
        rep_pre_trained = np.load(rep_knowledge_path)

        common_knowledge = []
        rep_knowledge = []
        demo_knowledge = []
        rep = 0
        demo = 0

        print('  - Reading Pre-trained Knowledge Embeddings...')
        for idx in range(vocab_size):
            mapping = 0
            for j, vocab_idx in enumerate(knowledge_indices['common']):
                if idx != 0 and idx == vocab_idx:
                    common_knowledge.append(common_pre_trained[j])
                    mapping = 1
                    break
            if mapping == 0:
                common_knowledge.append(np.zeros(embed_size))

            mapping = 0
            for j, vocab_idx in enumerate(knowledge_indices['rep']):
                if idx != 0 and idx == vocab_idx:
                    rep_knowledge.append(rep_pre_trained[j])
                    mapping = 1
                    rep += 1
                    break
            if mapping == 0:
                rep_knowledge.append(np.zeros(embed_size))

            mapping = 0
            for j, vocab_idx in enumerate(knowledge_indices['demo']):
                if idx != 0 and idx == vocab_idx:
                    demo_knowledge.append(demo_pre_trained[j])
                    mapping = 1
                    demo += 1
                    break
            if mapping == 0:
                demo_knowledge.append(np.zeros(embed_size))


        self.common_knowledge = nn.Embedding.from_pretrained(torch.FloatTensor(common_knowledge))
        self.demo_knowledge = nn.Embedding.from_pretrained(torch.FloatTensor(rep_knowledge))
        self.rep_knowledge = nn.Embedding.from_pretrained(torch.FloatTensor(demo_knowledge))

        self.fuse_knowledge_fc = nn.Linear(embed_size*2, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5
        self.fuse_knowledge_fc.weight.data.uniform_(-initrange, initrange)
        self.fuse_knowledge_fc.bias.data.zero_()

    def forward(self, word_embeddings: Tensor, texts: Tensor) -> Tensor:

        emb_with_ckwldg = (word_embeddings * self.alpha) + (self.common_knowledge(texts) * (1-self.alpha))

        demo_knwldg = (emb_with_ckwldg * self.beta) + (self.demo_knowledge(texts) * (1-self.beta))
        rep_knwldg = (emb_with_ckwldg * self.beta) + (self.rep_knowledge(texts) * (1-self.beta))

        #  concate and pass a FC layer
        emb_with_knowledge = self.fuse_knowledge_fc(torch.cat((demo_knwldg, rep_knwldg), 2))
        return self.dropout(emb_with_knowledge)

        #  return self.dropout(emb_with_ckwldg)



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


