import pandas as pd

# path = '/home/yyko/workspace/political_pre/KG_construction/Reddit_NF_liberal.csv'
path = '/home/yyko/workspace/political_pre/KG_construction/Reddit_NF_conservative.csv'
dest = '/home/yyko/workspace/political_pre/KG_construction'

# triplets.txt extraction
reddit_df = pd.read_csv(path, encoding='utf-8')

print("============================= Before Triplet Filtering ==============================")
print(len(reddit_df))

filtered_list = []

# 's 제거
# % 제거
# http 제거
# # 제거
# 'm 제거
# 길이가 1인 것 제거
# 소문자로 통일
for sen in reddit_df.filtered.values.tolist():
    if len(sen.split()[0]) != 1 and len(sen.split()[1]) != 1 and len(sen.split()[2]) != 1:
        filtered_list.append(sen.lower())

filtered_df = pd.DataFrame({'filtered' : filtered_list})

print("============================= After Triplet Filtering ==============================")
print(len(filtered_df))

# filtered_df.to_csv(dest + "/Reddit_NF_liberal_r.csv", index=False, encoding='utf-8-sig')
filtered_df.to_csv(dest + "/Reddit_NF_conservative_r.csv", index=False, encoding='utf-8-sig')

tab_split_list = []
for sen in filtered_df.filtered.values.tolist():
    tab_split_list.append('\t'.join(sen.split()))

# train : test = 8 : 2
train, test = tab_split_list[:int(len(tab_split_list)*0.8)], tab_split_list[int(len(tab_split_list)*0.8):]

# train : valid = 8 : 2
train, valid = train[:int(len(train)*0.8)], train[int(len(train)*0.8):]

print('train set: ' + str(len(train)))
print('valid set: ' + str(len(valid)))
print('test set: ' + str(len(test)))

# with open(dest + '/triplets_liberal.txt', 'w', encoding='UTF-8') as f:
#     for sen in tab_split_list:
#         f.write(sen + '\n')
# with open(dest + '/triplets_liberal_train.txt', 'w', encoding='UTF-8') as f:
#     for sen in train:
#         f.write(sen + '\n')
# with open(dest + '/triplets_liberal_valid.txt', 'w', encoding='UTF-8') as f:
#     for sen in valid:
#         f.write(sen + '\n')        
# with open(dest + '/triplets_liberal_test.txt', 'w', encoding='UTF-8') as f:
#     for sen in test:
#         f.write(sen + '\n')

with open(dest + '/triplets_conservative.txt', 'w', encoding='UTF-8') as f:
    for sen in tab_split_list:
        f.write(sen + '\n')
with open(dest + '/triplets_conservative_train.txt', 'w', encoding='UTF-8') as f:
    for sen in train:
        f.write(sen + '\n')
with open(dest + '/triplets_conservative_valid.txt', 'w', encoding='UTF-8') as f:
    for sen in valid:
        f.write(sen + '\n')        
with open(dest + '/triplets_conservative_test.txt', 'w', encoding='UTF-8') as f:
    for sen in test:
        f.write(sen + '\n')


print("=================================== Finished!!! ====================================")
