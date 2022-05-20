import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Hyperparameter setting
epoch = 10 
batch_size = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0005 # 0.0001 1e-4

class DataFrameDataset(data.Dataset):
    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.label 
            text = row.text
            examples.append(data.Example.fromlist([text, label], fields))
        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

def preprocess_text (text):
    text = text.str.lower() # lowercase
    text = text.str.replace(r"\#","") # replaces hashtags
    text = text.str.replace(r"http\S+","URL")  # remove URL addresses
    text = text.str.replace(r"@","")
    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    return text

class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)
    
    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0))
        x, _ = self.gru(x, h_0)
        h_t = x[:, -1, :]
        h_t = self.dropout(h_t)
        logit = self.out(h_t)
        return logit
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

def train(model, optimizer, train_iter):
    model.train()
    
    for batch in tqdm(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE) # x: [16, 1304] y: [16]
        y.data.sub_(1)
        optimizer.zero_grad()
        
        logit = model(x) # x: [16, 1304] logit: [16,3]
        
        loss = torch.nn.CrossEntropyLoss()(logit, y)
        loss.backward()
        optimizer.step()
        
def evaluate(model, val_iter):
    model.eval()
    corrects, total_loss = 0,0
    
    for batch in tqdm(val_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)
        logit = model(x)
        
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss/size
    avg_accuracy = 100.0 * corrects / size
    
    return avg_loss, avg_accuracy

if __name__ == "__main__":
    
    # To load dataset.csv
    dataset = pd.read_csv("news_dataset3.csv")
    dataset = dataset.drop(['ID', 'TITLE'], axis=1)

    # To pre-process text  
    dataset["text"]= preprocess_text(dataset["text"])
    
    # To define field format
    TEXT = data.Field(sequential=True, batch_first=True, lower=True)
    LABEL = data.Field(sequential=False, batch_first=True)  

    fields = [('text',TEXT), ('label',LABEL)]

    # To split the dataset
    train_df, valid_df = train_test_split(dataset)
    trainset, testset = DataFrameDataset.splits(fields, train_df=train_df, val_df=valid_df)
    
    MAX_VOCAB_SIZE = 25000
    
    # To create vocabulary (word embedding)
    TEXT.build_vocab(trainset, max_size = MAX_VOCAB_SIZE, 
                    vectors = 'glove.6B.200d',
                    unk_init = torch.Tensor.zero_)
    LABEL.build_vocab(trainset)
    
    # To create dataset: trainset, validationset, testset
    trainset, valset = trainset.split(split_ratio=0.8)
    
    train_iter, val_iter, test_iter = data.BucketIterator.splits((trainset,valset,testset), batch_size=batch_size)
    # iterator: text=['test','a', ... ] label = 'right'
    
    vocab_size = len(TEXT.vocab)
    n_classes = 3

    ####### MODEL: GRU #######
    model = GRU(1, 256, vocab_size, 200, n_classes, 0.2).to(DEVICE) # n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p
    # model.embed = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = None
    for e in range(1, epoch+1):
        
        train(model, optimizer, train_iter)
        val_loss, val_accuracy = evaluate(model, val_iter)
        print("[epoch: %d] val_loss:%5.3f, val_accuracy:%5.3f" % (e, val_loss, val_accuracy))
        
        if not best_val_loss or val_loss < best_val_loss:
            if not os.path.isdir("saves"):
                os.makedirs("saves")
            torch.save(model.state_dict(), './saves/txtclassification.pt')
            best_val_loss = val_loss

    model.load_state_dict(torch.load('./saves/txtclassification.pt'))
    test_loss, test_acc = evaluate(model, test_iter)
    print('test_loss:%5.3f, test_acc:%5.3f' % (test_loss, test_acc))
