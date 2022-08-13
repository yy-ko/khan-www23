import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize

raw_data_path = 'github/khan/data/khan_dataset.csv'
destination_folder = 'github/khan/data'

df_raw = pd.read_csv(raw_data_path)
# df_raw = df_raw.iloc[:1000,]

print(df_raw.head(10))
print(len(df_raw))
df_raw_r = df_raw.dropna(axis=0)
df_raw_r.reset_index(drop=True, inplace=True)
print(len(df_raw_r))
print(df_raw_r.head(10))

df_title = df_raw_r[['title']]
df_text = df_raw_r[['text']]
df_label = df_raw_r[['label']]

# df_title = df_title.dropna(axis=0)
# df_text = df_text.dropna(axis=0)
# df_label = df_label.dropna(axis=0)
# print(len(df_title))
# print(len(df_text))
# print(len(df_label))

sentences_list = []

sentences = df_text['text'].astype(str).apply(sent_tokenize)
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

df_all_1 = df_all.iloc[:100000,]
df_all_2 = df_all.iloc[100000:200000,]
df_all_3 = df_all.iloc[200000:300000,]
df_all_4 = df_all.iloc[300000:400000,]
df_all_5 = df_all.iloc[400000:500000,]
df_all_6 = df_all.iloc[500000:600000,]
df_all_7 = df_all.iloc[600000:700000,]
df_all_8 = df_all.iloc[700000:,]

df_all.to_csv(destination_folder + '/khan_dataset_SEP.csv', index=False, encoding='utf-8-sig')

df_all_1.to_csv(destination_folder + '/khan_split/khan_dataset_01.csv', index=False, encoding='utf-8-sig')
df_all_2.to_csv(destination_folder + '/khan_split/khan_dataset_02.csv', index=False, encoding='utf-8-sig')
df_all_3.to_csv(destination_folder + '/khan_split/khan_dataset_03.csv', index=False, encoding='utf-8-sig')
df_all_4.to_csv(destination_folder + '/khan_split/khan_dataset_04.csv', index=False, encoding='utf-8-sig')
df_all_5.to_csv(destination_folder + '/khan_split/khan_dataset_05.csv', index=False, encoding='utf-8-sig')
df_all_6.to_csv(destination_folder + '/khan_split/khan_dataset_06.csv', index=False, encoding='utf-8-sig')
df_all_7.to_csv(destination_folder + '/khan_split/khan_dataset_07.csv', index=False, encoding='utf-8-sig')
df_all_8.to_csv(destination_folder + '/khan_split/khan_dataset_08.csv', index=False, encoding='utf-8-sig')
# df_all.to_csv(destination_folder + '/test.csv', index=False, encoding='utf-8-sig')


