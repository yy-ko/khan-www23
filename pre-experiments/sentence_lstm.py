import os
import numpy as np
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

raw_data_path = '/home/yyko/workspace/political_pre/news_dataset_test.csv'
destination_folder = '/home/yyko/workspace/political_pre'

print('STEP-1: successfully completed DATA LOADING\n')

stop_words = stopwords.words('english')    

df_raw = pd.read_csv(raw_data_path)

df_raw['titletext'] = df_raw['title'] + '. ' + df_raw['text']
df_raw = df_raw[['titletext', 'label']]
df_raw.columns = ['text', 'label']

df_left = df_raw[df_raw['label'] == 0]
df_center = df_raw[df_raw['label'] == 1]
df_right = df_raw[df_raw['label'] == 2]

train_test_ratio = 0.80

df_left_train, df_left_test = train_test_split(df_left, train_size = train_test_ratio, random_state = 1)
df_center_train, df_center_test = train_test_split(df_center, train_size = train_test_ratio, random_state = 1)
df_right_train, df_right_test = train_test_split(df_right, train_size = train_test_ratio, random_state = 1)

df_train = pd.concat([df_left_train, df_center_train, df_right_train], ignore_index=True, sort=False)
df_test = pd.concat([df_left_test, df_center_test, df_right_test], ignore_index=True, sort=False)
df_raw = pd.concat([df_train, df_test], ignore_index=True, sort=False)

print('STEP-2: successfully completed DATASET BALANCED\n')

'''
# 토큰화 함수
def tokenization(sentences):
    return [word_tokenize(sentence) for sentence in sentences]

# 전처리 함수
def preprocess_sentence(sentence):
    # 영어를 제외한 숫자, 특수 문자 등은 전부 제거. 모든 알파벳은 소문자화
    sentence = [re.sub(r'[^a-zA-z\s]', '', word).lower() for word in sentence]
    # 불용어가 아니면서 단어가 실제로 존재해야 함
    return [word for word in sentence if word not in stop_words and word]

# 위 전처리 함수를 모든 문장에 대해서 수행
def preprocess_sentences(sentences):
    return [preprocess_sentence(sentence) for sentence in sentences]

# 문장 하나는 50차원의 벡터 임베딩값
embedding_dim = 50
zero_vector = np.zeros(embedding_dim)

# 문장 내에 있는 단어 벡터들의 평균을 구하는 함수 정의(aggregation)
# 문장 길이가 0일 경우에는 50차원의 영벡터를 return
def calculate_sentence_vector(sentence):
    if len(sentence) != 0:
        z = sum([glove_dict.get(word, zero_vector) for word in sentence]) / len(sentence)
        return z.tolist()
    else:
        return zero_vector

### 문장 하나에 대한 단어 벡터를 반환하고 각 문장을 GloVe => embedding > aggr(mean, lstm) > classification
# 각 문장에 대해 문장 벡터를 반환 (문장 하나 => 벡터 하나(50차원))
def sentences_to_vectors(sentences):
    return [calculate_sentence_vector(sentence) for sentence in sentences]
'''

sentences=df_raw['text'].astype(str).apply(sent_tokenize)
print('STEP-3: successfully completed SENTENCE TOKENIZATION\n')

'''
# 토큰화된 sentence의 값들을 list로 저장
s_tokenize=sentences.values.tolist()

# 토큰화된 sentence의 각 sentence들을 word 단위로 토큰화
w_tokenize=list(map(tokenization,s_tokenize))

# word들에 대한 전처리 수행
prepro = list(map(preprocess_sentences,w_tokenize))

# sentence-level embedding aggregation
# : sentence 속 각 word들의 embedding 값들을 aggr(mean)해서 하나의 sentence를 표현하는 embedding vector로 변환
sentence_embed=list(map(sentences_to_vectors,prepro))
'''

# model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
# model = SentenceTransformer('all-mpnet-base-v2', device=device)
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

print("Max Sequence Length:", model.max_seq_length)
#Change the length to 100
model.max_seq_length = 100
print("Max Sequence Length:", model.max_seq_length)

def sentence_embedding(sentences):
    return [model.encode(sentence, device=device).tolist() for sentence in sentences]

sentence_embed = sentence_embedding(sentences)

df_raw['s_embed']=s_embed
df_raw = df_raw[['label', 's_embed']]
df_raw = df_raw.reindex(columns = ['label', 's_embed'])
print(df_raw.head())

texts = list(df_raw['s_embed'])
labels = list(df_raw['label'])

print('STEP-4: successfully completed SENTENCE-LEVEL EMBEDDING\n')

# Fixed the number of sentence embedding vectors in a row to 64
def get_vector(s):
    while len(s) < 64: 
        s.append([0] * 384)
    return s[:64]

sentence_embed = [get_vector(s) for s in sentence_embed]
print('STEP-5: successfully completed SENTENCE-LEVEL SHAPE TRANSFORM\n')

r = len(sentence_embed)*0.8
x_train, x_test = np.array(sentence_embed[:int(r)], dtype=object), np.array(sentence_embed[int(r):], dtype=object)
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
y = [label for label in labels]

y_train, y_test = np.array(y[:int(r)]), np.array(y[int(r):])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# batch_size = 16
total_epochs = 20
# train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_data = train_data.repeat().batch(batch_size, drop_remainder=True)
# steps_per_epoch = len(x_train) // batch_size

print('STEP-6: successfully completed BATCH-SIZE CONFIGURATION\n')

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))) # (batch_size, 64, 384)
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(192))) # (batch_size, 64, 384)
model.add(tf.keras.layers.Dropout(0.2)) # (batch_size, 64, 384)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(64, activation = 'relu')) # (batch_size, 64, 64)
model.add(tf.keras.layers.Dense(3, activation = 'softmax')) # (batch_size, 64, 3)
print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('STEP-7: MODEL TRAINING START\n')

x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')

model.fit(x_train, y_train, epochs = total_epochs, validation_data=(x_test, y_test))
print()
model.evaluate(x_test, y_test, verbose=2)
 
