import pandas as pd
from itertools import islice
from tqdm import tqdm

# path = '/home/yyko/workspace/political_pre/KG_construction/Reddit_NF_liberal_r.csv'
path = '/home/yyko/workspace/political_pre/KG_construction/Reddit_NF_conservative_r.csv'
dest = '/home/yyko/workspace/political_pre/KG_construction'

reddit_df = pd.read_csv(path, encoding='utf-8')
# reddit_df = reddit_df.iloc[:100,]

head_list = []
tail_list = []
entity_list = []
relation_list = []

for sen in tqdm(reddit_df.filtered.values.tolist()):
    head_list.append(sen.split()[0])
print(len(head_list))

for sen in tqdm(reddit_df.filtered.values.tolist()):
    try:
        tail_list.append(sen.split()[2])
    except:
        print("============================ignored the exception============================")
        pass
print(len(tail_list))

entity_list = head_list + tail_list
entity_list2 = list(set(entity_list))
entity_dict = {i : entity_list2[i] for i in range(len(entity_list2))}
print(len(entity_dict))

for sen in tqdm(reddit_df.filtered.values.tolist()):
    try:
        relation_list.append(sen.split()[1])
    except:
        print("============================ignored the exception============================")
        pass
print(len(relation_list))

relation_list2 = list(set(relation_list))
relation_dict = {i : relation_list2[i] for i in range(len(relation_list2))}
print(len(relation_dict))  

print(dict(islice(entity_dict.items(), 10)))
print(dict(islice(relation_dict.items(), 10)))

# with open(dest + '/entities_liberal.dict','w',encoding='UTF-8') as f:
#     for key,value in entity_dict.items():
#         f.write(f'{key}\t{value}\n')

# with open(dest + '/relations_liberal.dict','w',encoding='UTF-8') as f:
#     for key,value in relation_dict.items():
#         f.write(f'{key}\t{value}\n')

with open(dest + '/entities_conservative.dict','w',encoding='UTF-8') as f:
    for key,value in entity_dict.items():
        f.write(f'{key}\t{value}\n')
        
with open(dest + '/relations_conservative.dict','w',encoding='UTF-8') as f:
    for key,value in relation_dict.items():
        f.write(f'{key}\t{value}\n')
