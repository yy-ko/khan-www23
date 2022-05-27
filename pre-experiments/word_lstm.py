import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
# pip install -U torch==1.8.0 torchtext==0.9.0
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
# import torchtext.legacy
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

raw_data_path = '/home/yyko/workspace/political_pre/news_dataset3_r.csv'
destination_folder = '/home/yyko/workspace/political_pre'

train_test_ratio = 0.90
train_valid_ratio = 0.80
first_n_words = 25000

def trim_string(x):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(x)
    result = []
    for word in word_tokens: 
        if word not in stop_words: 
            result.append(word)
    result = ' '.join(result[:first_n_words])
    return result

df_raw = pd.read_csv(raw_data_path)

df_raw.drop(df_raw[df_raw.text.str.len()<5].index, inplace = True)

df_raw['titletext'] = df_raw['title'] + '.' + df_raw['text']
df_raw = df_raw[['label','titletext']]
df_raw = df_raw.reindex(columns = ['label', 'titletext'])

# df_raw['text'] = df_raw['text'].astype(str).apply(trim_string)
df_raw['titletext'] = df_raw['titletext'].astype(str).apply(trim_string)
df_raw['titletext'] = df_raw['titletext'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
def one_space(a):
    return ' '.join(a.split())
df_raw['titletext'] = df_raw['titletext'].apply(one_space)

df_left = df_raw[df_raw['label'] == 0]
df_center = df_raw[df_raw['label'] == 1]
df_right = df_raw[df_raw['label'] == 2]


df_left_full_train, df_left_test = train_test_split(df_left, train_size = train_test_ratio, random_state = 1)
df_center_full_train, df_center_test = train_test_split(df_center, train_size = train_test_ratio, random_state = 1)
df_right_full_train, df_right_test = train_test_split(df_right, train_size = train_test_ratio, random_state = 1)


df_left_train, df_left_valid = train_test_split(df_left_full_train, train_size = train_valid_ratio, random_state = 1)
df_center_train, df_center_valid = train_test_split(df_center_full_train, train_size = train_valid_ratio, random_state = 1)
df_right_train, df_right_valid = train_test_split(df_right_full_train, train_size = train_valid_ratio, random_state = 1)


df_train = pd.concat([df_left_train, df_center_train, df_right_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_left_valid, df_center_valid, df_right_valid], ignore_index=True, sort=False)
df_test = pd.concat([df_left_test, df_center_test, df_right_test], ignore_index=True, sort=False)


df_train.to_csv(destination_folder + '/train.csv', index=False, encoding='utf-8-sig')
df_valid.to_csv(destination_folder + '/valid.csv', index=False, encoding='utf-8-sig')
df_test.to_csv(destination_folder + '/test.csv', index=False, encoding='utf-8-sig')

# Fields
label_field = Field(sequential = False,
                    use_vocab = False,
                    batch_first = True,
                    dtype = torch.float) 
text_field = Field(tokenize = 'spacy', 
                   use_vocab = True,
                   lower = True, 
                   include_lengths = True, 
                   batch_first = True)
fields = [('label', label_field), ('titletext', text_field)]


train, valid, test = TabularDataset.splits(path=destination_folder, train='train.csv', validation='valid.csv', test='test.csv',
                                           format='CSV', fields=fields, skip_header=True)
print('############################################')
print('train 데이터 개수 : {}'.format(len(train)))
print('valid 데이터 개수 : {}'.format(len(valid)))
print('test 데이터 개수 : {}'.format(len(test)))
print('############################################')
print(vars(test[0]))

# Iterators
train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.titletext),
                            device=device, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.titletext),
                            device=device, sort=True, sort_within_batch=True)
test_iter = BucketIterator(test, batch_size=16, sort_key=lambda x: len(x.titletext),
                            device=device, sort=True, sort_within_batch=True)

# Vocabulary
text_field.build_vocab(train, min_freq=3)
class LSTM(nn.Module): 

    def __init__(self, dimension = 256):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 128)
        self.dimension = dimension 
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True) 
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(2*dimension, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text, text_len):
        text_emb = self.embedding(text)
        packed_input = pack_padded_sequence(text_emb, 
                                            text_len.cpu(), 
                                            batch_first=True, 
                                            enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first = True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1) #[range(len(output)), text_len -1 + 0 , self.dimension]
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        # text_fea = torch.squeeze(text_fea, 1)
        # softmax function
        softmax_out = self.softmax(text_fea)
        # reshape to be batch_size first
        softmax_out = softmax_out.view(softmax_out.size(0), -1, 3)
        softmax_out = softmax_out[:, -1]

        return softmax_out

def save_checkpoint(save_path, model, optimizer, valid_loss):

    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model, optimizer):

    if load_path==None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

def train(model,
          optimizer,
          criterion = nn.CrossEntropyLoss(),
          train_loader = train_iter,
          valid_loader = valid_iter,
          num_epochs = 10,
          eval_every = len(train_iter) // 2,
          file_path = destination_folder,
          best_valid_loss = float("Inf")):

    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    
    model.train()
    for epoch in tqdm(range(num_epochs)) :
        for (labels, (titletext, titletext_len)), _ in train_loader:
            labels = torch.tensor(labels, dtype=torch.long, device=device)
    
            titletext = titletext.to(device)
            titletext_len = titletext_len.to(device)
            
            output = model(titletext, titletext_len)

            loss = criterion(output, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    for (labels, (titletext, titletext_len)), _ in valid_loader:
                        labels = torch.tensor(labels, dtype=torch.long, device=device)
                        titletext = titletext.to(device)
                        titletext_len = titletext_len.to(device)
                        output = model(titletext, titletext_len)

                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

model = LSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model=model, optimizer=optimizer, num_epochs=20)


# EVALUATION
print('****************************** MODEL EVALUATION ****************************************')

train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + '/metrics.pt')
print('go go go')

plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('wd_lstm_result.png')

def evaluate(model, test_loader, threshold=0.2):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels, (titletext, titletext_len)), _ in test_loader:           
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            titletext = titletext.to(device)
            titletext_len = titletext_len.to(device)
            output = model(titletext, titletext_len)
            output = (output > threshold).int()        
                       
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())
        
        y_pred =np.argmax(y_pred, axis=1)
        
    print('*********************************************')
    print()
    print('Confusion Matrix')
    ryu = multilabel_confusion_matrix(y_true, y_pred, labels=[0,1,2])
    tn0, fp0, fn0, tp0 = ryu[0].ravel()
    tn1, fp1, fn1, tp1 = ryu[1].ravel()
    tn2, fp2, fn2, tp2 = ryu[2].ravel()
    
    recall0 = tp0/(tp0 + fn0) 
    precision0 = tp0/(tp0 + fp0)
    recall1 = tp1/(tp1 + fn1)
    precision1 = tp1/(tp1 + fp1)
    recall2 = tp2/(tp2 + fn2) 
    precision2 = tp2/(tp2 + fp2)
    
    accuracy0 = (tp0 + tn0)/(tp0 + tn0 + fp0 + fn0)
    accuracy1 = (tp1 + tn1)/(tp1 + tn1 + fp1 + fn1)
    accuracy2 = (tp2 + tn2)/(tp2 + tn2 + fp2 + fn2)
    
    f1score0 = 2*recall0*precision0/(recall0 + precision0)
    f1score1 = 2*recall1*precision1/(recall1 + precision1)
    f1score2 = 2*recall2*precision2/(recall2 + precision2)
    
    print('tn: true negative')
    print('fp: false positive')
    print('fn: false negative')
    print('tp: true positive')
    print('0:left, 1:center, 2:right')
    print()
    print('label0: tn, fp, fn, tp', tn0, fp0, fn0, tp0)
    print('label1: tn, fp, fn, tp', tn1, fp1, fn1, tp1)
    print('label2: tn, fp, fn, tp', tn2, fp2, fn2, tp2)
    print()
    
    print('label0: recall {:.4f} precision {:.4f}'.format(recall0, precision0))
    print('label1: recall {:.4f} precision {:.4f}'.format(recall1, precision1))
    print('label2: recall {:.4f} precision {:.4f}'.format(recall2, precision2))
    print()
    print('*********************************************')
    print('left: Accuracy {:.4f} f1-score {:.4f}'.format(accuracy0, f1score0))
    print('center: Accuracy {:.4f} f1-score {:.4f}'.format(accuracy1, f1score1))
    print('right: Accuracy {:.4f} f1-score {:.4f}'.format(accuracy2, f1score2))
    # print('Average Accuracy: {:.4f}'.format((accuracy0 + accuracy1 + accuracy2)/3))
    # print('Average F1-score: {:.4f}'.format((f1score0 + f1score1 + f1score2)/3))
    print()
    print('Here is the accuracy of word-level model:LSTM')
    print('*********************************************')
    print(classification_report(y_true, y_pred))


best_model = LSTM().to(device)
optimizer = optim.Adam(best_model.parameters(), lr=0.001)
load_checkpoint(destination_folder + '/model.pt', best_model, optimizer)
evaluate(best_model, test_iter)
