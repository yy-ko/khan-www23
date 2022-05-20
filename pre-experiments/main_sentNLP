import os
from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
import pandas as pd
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoTokenizer, AutoModel

# Hyperparameter setting
epoch = 10 
batch_size = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0005 # 0.0001 1e-4

n_sentences = 20
n_words = 70

def preprocess_text (text):
    text = text.lower() # lowercase
    text = text.replace(r"\#","") # replaces hashtags
    text = text.replace(r"http\S+","URL")  # remove URL addresses
    text = text.replace(r"@","")
    text = text.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.replace("\s{2,}", " ")
    return text
    
class Classifier(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_sentences, embed_dim, n_classes, dropout_p=0.2): 
        super(Classifier, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.LSTM(embed_dim, self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)
        
    # def forward(self, x):
    def forward(self, input_text, mask):
        batch_size = input_text.size(0)
        input_text = input_text.view(-1, n_words) # [160,20]
        mask = mask.view(-1, n_words)
                
        model_output = s_bert(input_ids=input_text, attention_mask=mask).last_hidden_state           # model_output: [160,20,384]
        sentence_embeddings = mean_pooling(model_output, mask)                                       # sentence_embeddings: [160,384]
        sentence_embeddings = sentence_embeddings.view(batch_size, -1, sentence_embeddings.size(-1)) # sentence_embeddings: [16,10,384]
        
        # GRU
        h_t, _ = self.gru(sentence_embeddings) # [16, 10, 256]
        h_t = h_t[:, -1, :] # [16, 256]
        h_t = self.dropout(h_t)
        logit = self.out(h_t)
        
        return logit
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

def train(model, optimizer, train_iter):
    model.train()   
    
    for batch in tqdm(train_iter):
        input_text, mask, y = batch
        
        input_text = input_text.to(DEVICE).long() # [16,10,20]
        mask = mask.to(DEVICE).long() # [16,10,20]
        y = y.to(DEVICE).long()
        
        optimizer.zero_grad()
        logit = model(input_text, mask) # sentence_embeddings

        loss = torch.nn.CrossEntropyLoss()(logit, y)
        loss.backward()
        optimizer.step()
        
        
def evaluate(model, val_iter):
    model.eval()
    corrects, total_loss = 0,0
    
    for batch in tqdm(val_iter):
        input_text, mask, y = batch
        
        input_text = input_text.to(DEVICE).long() # [16,10,20]
        mask = mask.to(DEVICE).long()
        y = y.to(DEVICE).long()
        
        logit = model(input_text, mask) # sentence_embeddings
        
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss/size
    avg_accuracy = 100.0 * corrects / size
    
    return avg_loss, avg_accuracy

class CustomDataset(data.Dataset):
    def __init__(self, df, fields, s_tokenizer, n_sentences, n_words, is_test=False, **kwargs):
        self.s_tokenizer = s_tokenizer
        self.examples = []
        self.n_sentences = n_sentences
        self.n_words = n_words
        
        for i, row in df.iterrows():
            label = row.label 
            text = row.text
            self.examples.append(data.Example.fromlist([text, label], fields))
    
    def __getitem__(self, i):
        row = self.examples[i]
        text, label = row.text, row.label
        # text = _process(text, self.n_sentences)
        tokenized_result = self.s_tokenizer(text[:self.n_sentences], truncation=True, return_tensors='pt', 
                                            max_length=self.n_words, padding='max_length', add_special_tokens=False)
        
        inputs = tokenized_result['input_ids']
        mask = tokenized_result['attention_mask']

        if inputs.size(0) <= self.n_sentences:
            zeros = torch.zeros(self.n_words)
            zeros = zeros.unsqueeze(0).repeat(self.n_sentences-inputs.size(0), 1)
            inputs = torch.cat([inputs, zeros], dim=0)
            mask = torch.cat([mask, zeros], dim=0)
        
        if label == 'left':
            label = torch.tensor(0, dtype=torch.long)
        elif label == 'center':
            label = torch.tensor(1, dtype=torch.long)
        else:
            label = torch.tensor(2, dtype=torch.long)
            
        return inputs.long(), mask.long(), label

def tokenizer(text):
    sent_zip = sent_tokenize(text)
    for i, sentence in enumerate(sent_zip): 
        sent_zip[i] = preprocess_text(sentence)
        
    return sent_zip

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

if __name__ == "__main__":
    file_name = "news_dataset3.csv"
    
    # To load dataset.csv
    dataset = pd.read_csv(file_name)
    dataset = dataset.drop(['ID', 'TITLE'], axis=1)
    
    # To define field format
    TEXT = data.Field(tokenize = tokenizer, sequential=True, batch_first=True, lower=True)
    LABEL = data.Field(sequential=False, batch_first=True)  
    
    fields = [('text',TEXT), ('label',LABEL)]

    s_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
    s_bert = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2').to(DEVICE)
    # s_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # s_bert = AutoModel.from_pretrained('bert-base-uncased')
    
    # To split the dataset
    train_df, test_df = train_test_split(dataset, test_size=0.25, stratify = dataset['label'], random_state=42)
    train_df, valid_df = train_test_split(train_df, test_size=0.20, random_state=42)
    
    # Dataset: set of examples text: sentences / label: right, center, left
    trainset = CustomDataset(train_df, fields, s_tokenizer, n_sentences, n_words)
    valset = CustomDataset(valid_df, fields, s_tokenizer, n_sentences, n_words)
    testset = CustomDataset(test_df, fields, s_tokenizer, n_sentences, n_words)

    # DataLoader
    train_iter = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    
    # hyper parameter set-up
    n_classes = 3
    embed_dim = 384
    hidden_dim = 256
    dropout_p = 0.2
    n_layers = 1
    
    ####### MODEL #######
    model = Classifier(n_layers, hidden_dim, n_sentences, embed_dim, n_classes, dropout_p).to(DEVICE) 
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
    
