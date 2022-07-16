import pandas as pd

path = '/home/yyko/workspace/political_pre/KG_construction/Reddit_result_conservative.csv'
dest = '/home/yyko/workspace/political_pre/KG_construction'

reddit_df = pd.read_csv(path, encoding='utf-8')

print("============================= Before Triplet Filtering ==============================")
print(len(reddit_df))

filtered_list = []

for sen in reddit_df.triplet.values.tolist():
  if len(sen.split()) == 3:
    filtered_list.append(sen)
    
filtered_df = pd.DataFrame({'filtered' : filtered_list})

print("============================= After Triplet Filtering ==============================")
print(len(filtered_df))

filtered_df.to_csv(dest + "/Reddit_NF_con_01.csv", index=False, encoding='utf-8-sig')

print("=================================== Finished!!! ====================================")
