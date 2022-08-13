import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize

raw_data_path = '/home/yyko/workspace/political_pre/github/khan/data/semeval.csv'
destination_folder = '/home/yyko/workspace/political_pre/github/khan/data'

df_raw = pd.read_csv(raw_data_path)

print(df_raw.head(10))
print(len(df_raw))
df_raw_r = df_raw.dropna(axis=0)
df_raw_r.reset_index(drop=True, inplace=True)
print(len(df_raw_r))
print(df_raw_r.head(10))

df_title = df_raw_r[['title']]
df_text = df_raw_r[['text']]
df_label = df_raw_r[['label']]

sentences_list = []
remove_sep = []

for index, row in df_text.iterrows():
    remove_str = row.text.replace('<SEP>', '')
    remove_sep.append(remove_str)

df_text_remove_SEP = pd.DataFrame(remove_sep, columns=['text'])

sentences = df_text_remove_SEP['text'].astype(str).apply(sent_tokenize)
for row_sens in sentences:
    # print(len(row_sens))
    sen_join = ' <SEP> '.join(sen for sen in row_sens)
    sentences_list.append(sen_join)

df_text = pd.DataFrame(sentences_list, columns=['text'])
# df_text.columns = ["text"]
print(len(df_title))
print(len(df_text))
print(len(df_label))

df_all = pd.concat([df_title, df_text, df_label], axis = 1)
# df_all = df_all.dropna()
print(len(df_all))
print(df_all.head(10))

df_all.to_csv(destination_folder + '/semeval_dataset_SEP.csv', index=False, encoding='utf-8-sig')
