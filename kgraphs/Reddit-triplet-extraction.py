import pandas as pd
import numpy as np
from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser
from nltk.tree import ParentedTree
import os
import torch
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

# liberal: 18만 data
# Conservative: 18만 data
raw_data_path = '/home/yyko/workspace/political_pre/KG_construction/reddit_conservative_v3.csv'
destination_folder = '/home/yyko/workspace/political_pre/KG_construction'

reddit_df = pd.read_csv(raw_data_path, encoding='utf-8')
reddit_df = reddit_df[['body']]

# reddit_df = reddit_df.iloc[120000:,]
sentences=reddit_df['body'].astype(str).apply(sent_tokenize)

# sentence split from text
all=[]
for i in range(len(sentences)):
    all.extend(sentences.values[i])
reddit_sen = pd.DataFrame(all)

reddit_sen.columns = ['sentence']

print("============================= Sentence Length ==============================")
print(len(reddit_sen))

# # stopwords preprocessing
# first_n_words = 1000
# def trim_string(x):
#     stop_words = set(stopwords.words('english'))
#     word_tokens = word_tokenize(x)
#     result = []
#     for word in word_tokens: 
#         if word not in stop_words: 
#             result.append(word)
#     # x = x.split(maxsplit=first_n_words)
#     result = ' '.join(result[:first_n_words])
#     return result

# reddit_sen['sentence'] = reddit_sen['sentence'].astype(str).apply(trim_string)
# reddit_sen['sentence'] = reddit_sen['sentence'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)

print("=============================== Reddit Data ================================")
print(reddit_sen.head(10))
print('')

# StanfordCoreNLPServer connection (start server)
# cd stanford-corenlp-4.4.0                                                                                
# java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000

# Shutdown (end server)
# wget "localhost:9000/shutdown?key=`cat /tmp/corenlp.shutdown`" -O -
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')

triplet_result = []

def triplet_extraction (input_sent, output=[
                                            # 'parse_tree',
                                            # 'spo',
                                            'result'
                                            ]):
    # Parse the input sentence with Stanford CoreNLP Parser
    # pos_type = pos_tagger.tag(input_sent.split())
    parse_tree, = ParentedTree.convert(list(pos_tagger.parse(input_sent.split()))[0])
    dep_type, = ParentedTree.convert(dep_parser.parse(input_sent.split()))
    # Extract subject, predicate and object
    subject = extract_subject(parse_tree)
    predicate = extract_predicate(parse_tree)
    objects = extract_object(parse_tree)
    if 'parse_tree' in output:
        print('---Parse Tree---')
        parse_tree.pretty_print()
    if 'spo' in output:
        print('---Subject---')
        print(subject)
        print('---Predicate---')
        print(predicate)
        print('---Object---')
        print(objects)
    if 'result' in output:
        # print('---Result---')
        res = ' '.join([subject[0], predicate[0], objects[0]])
        print(res)
        triplet_result.append(res)
        

def extract_subject (parse_tree):
    # Extract the first noun found in NP_subtree
    subject = []
    for s in parse_tree.subtrees(lambda x: x.label() == 'NP'):
        for t in s.subtrees(lambda y: y.label().startswith('NN')):
            output = [t[0], extract_attr(t)]
            # Avoid empty or repeated values
            if output != [] and output not in subject:
                subject.append(output) 
    if len(subject) != 0: return subject[0] 
    else: return ['']

def extract_predicate (parse_tree):
    # Extract the deepest(last) verb foybd ub VP_subtree
    output, predicate = [],[]
    for s in parse_tree.subtrees(lambda x: x.label() == 'VP'):
        for t in s.subtrees(lambda y: y.label().startswith('VB')):
            output = [t[0], extract_attr(t)]
            if output != [] and output not in predicate:    
                predicate.append(output)
    if len(predicate) != 0: return predicate[-1]
    else: return ['']

def extract_object (parse_tree):
    # Extract the first noun or first adjective in NP, PP, ADP siblings of VP_subtree
    objects, output, word = [],[],[]
    for s in parse_tree.subtrees(lambda x: x.label() == 'VP'):
        for t in s.subtrees(lambda y: y.label() in ['NP','PP','ADP']):
            if t.label() in ['NP','PP']:
                for u in t.subtrees(lambda z: z.label().startswith('NN')):
                    word = u          
            else:
                for u in t.subtrees(lambda z: z.label().startswith('JJ')):
                    word = u
            if len(word) != 0:
                output = [word[0], extract_attr(word)]
            if output != [] and output not in objects:
                objects.append(output)
    if len(objects) != 0: return objects[0]
    else: return ['']

def extract_attr (word):
    attrs = []     
    # Search among the word's siblings
    if word.label().startswith('JJ'):
        for p in word.parent(): 
            if p.label() == 'RB':
                attrs.append(p[0])
    elif word.label().startswith('NN'):
        for p in word.parent():
            if p.label() in ['DT','PRP$','POS','JJ','CD','ADJP','QP','NP']:
                attrs.append(p[0])
    elif word.label().startswith('VB'):
        for p in word.parent():
            if p.label() == 'ADVP':
                attrs.append(p[0])
    # Search among the word's uncles
    if word.label().startswith('NN') or word.label().startswith('JJ'):
        for p in word.parent().parent():
            if p.label() == 'PP' and p != word.parent():
                attrs.append(' '.join(p.flatten()))
    elif word.label().startswith('VB'):
        for p in word.parent().parent():
            if p.label().startswith('VB') and p != word.parent():
                attrs.append(' '.join(p.flatten()))
    return attrs

print("======================== Triplet Extraction Example ========================")

triplet_extraction('A rare black squirrel has become a regular visitor to a suburban garden')
print('')

print("========================= Reddit Triplet Extraction ========================")
 
try:
    for sen in tqdm(reddit_sen.sentence.values.tolist()):
        try:
            triplet_extraction(sen).device()
        except:
            print("============================ignored the exception============================")
            pass
except Exception:
    pass

df_triplet = pd.DataFrame(triplet_result)
df_triplet.columns = ['triplet']
df_triplet.to_csv(destination_folder + "/Reddit_result_conservative_3.csv", index=False, encoding='utf-8-sig')

print(df_triplet.head(10))
