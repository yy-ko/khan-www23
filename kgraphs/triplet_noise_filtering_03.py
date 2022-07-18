# Named Entity Recognition (NER) apply to Reddit triplets

###################################### NER - Flair ######################################
# from flair.data import Sentence
# from flair.models import SequenceTagger

# # load tagger
# tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

# # make example sentence
# # sentence = Sentence("On September 1st George won 1 dollar while watching Game of Thrones.")
# sentence = Sentence("Google LLC is a multinational technology company. It was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in Califonia.")

# # predict NER tags
# tagger.predict(sentence)

# # print sentence
# print(sentence)

# # print predicted NER spans
# print('The following NER tags are found:')
# # iterate over entities and print
# for entity in sentence.get_spans('ner'):
#     print(entity)

###################################### NER - SpaCy ######################################
import spacy
import en_core_web_sm
nlp = spacy.load("en_core_web_sm")
# article = nlp("Google LLC is a multinational technology company. It was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in Califonia.")
# article = nlp("President Joe Biden departs Holy Trinity Catholic Church in the Georgetown section of Washington, after attending a Mass in Washington on July 17, 2022.")

# for X in article.ents:
#     print("Text:", X.text, "\tLabel:", X.label_)

###################################### NER - Reddit #####################################
import pandas as pd
from tqdm import tqdm

# path = '/home/yyko/workspace/political_pre/KG_construction/Reddit_NF_liberal.csv'
path = '/home/yyko/workspace/political_pre/KG_construction/Reddit_NF_conservative.csv'
dest = '/home/yyko/workspace/political_pre/KG_construction'

reddit_df = pd.read_csv(path, encoding='utf-8')

print("============================= Before Triplet Filtering ==============================")
print(len(reddit_df))

filtered_list = []
for sen in tqdm(reddit_df.filtered.values.tolist()):
    triplet = nlp(sen)
    for x in triplet.ents:
        if x.label_ == 'PERSON' or x.label_ == 'ORG' or x.label_ == 'NORP' or x.label_ == 'EVENT' or x.label_ == 'GPE' or x.label_ == 'LOC':
            print(sen)
            filtered_list.append(sen)

filtered_list_deduplication = list(set(filtered_list))
filtered_df = pd.DataFrame({'filtered' : filtered_list_deduplication})

print("============================= After Triplet Filtering ==============================")
print(len(filtered_df))

# filtered_df.to_csv(dest + "/Reddit_NF_FINAL_liberal.csv", index=False, encoding='utf-8-sig')
filtered_df.to_csv(dest + "/Reddit_NF_FINAL_conservative.csv", index=False, encoding='utf-8-sig')

print("=================================== Finished!!! ====================================")
