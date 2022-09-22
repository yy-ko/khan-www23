from pickle import TRUE
import torch, logging, sys
from torch.utils.data import dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

# from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset

from collections import Counter, OrderedDict
from typing import Iterable
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


MAX_WORDS = 40

def yield_tokens(data_iter, tokenizer):
    for _, title, text in data_iter:
        yield tokenizer(title)
        yield tokenizer(text)

def preprocess_text(text):
    text = text.str.lower() # lowercase
    text = text.str.replace(r"\#","") # replaces hashtags
    text = text.str.replace(r"http\S+","")  # remove URL addresses
    #  text = text.str.replace(r"@","")
    #  text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    #  text = text.str.replace("\s{2,}", " ")
    return text

def data_preprocessing(dataset, data_path):
    """
        Args:
            dataset:
            data_path:
        Returns:
            vocab_size:
            num_class:
    """

    num_class = 0

    if dataset =='SEMEVAL':
        num_class = 2
        tokenizer = get_tokenizer('basic_english')
        #  data_path += '/semeval.csv'
        data_path += '/semeval-new.csv'

    elif dataset =='ALLSIDES-S':
        num_class = 3
        data_path += '/khan_dataset.csv'

    elif dataset =='ALLSIDES-L':
        num_class = 5
        data_path += '/khan_dataset.csv'

    else:
        logging.error('Invalid dataset name!')
        sys.exit(1)

    # read a dataset file from a local path and pre-process it
    dataset = pd.read_csv(data_path)
    #  dataset["text"] = preprocess_text(dataset["text"].astype(str))
    dataset = dataset[['text','title','label']]

    # split a dataset into train/test datasets
    train_df, test_df = train_test_split(dataset, train_size=0.9, shuffle=TRUE)
    v_train = train_df.values
    v_test = test_df.values

    train_iter = list(map(lambda x: (x.tolist()[2], x.tolist()[1], x.tolist()[0]), v_train))
    test_iter = list(map(lambda x: (x.tolist()[2], x.tolist()[1], x.tolist()[0]), v_test))

    # build vocab
    tokenizer = get_tokenizer('basic_english')
    #  vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=['<unk>', '<splt>'])
    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=['<unk>', '<sep>'])
    vocab.set_default_index(vocab['<unk>'])

    knowledge_indices = {}
    rep_entity_list = []
    demo_entity_list = []
    common_entity_list = []

    with open('./kgraphs/pre-trained/entities_con.dict') as rep_file:
        while (line := rep_file.readline().rstrip()):
            rep_entity_list.append(line.split()[1])

    with open('./kgraphs/pre-trained/entities_lib.dict') as demo_file:
        while (line := demo_file.readline().rstrip()):
            demo_entity_list.append(line.split()[1])

    with open('./kgraphs/pre-trained/entities_yago.dict') as rep_file:
        while (line := rep_file.readline().rstrip()):
            #  print (line.split()[1].split('_')[0].lower())
            common_entity_list.append(line.split()[1].split('_')[0].lower())

    #  with open('./kgraphs/pre-trained/entities_FB15K.dict') as rep_file:
        #  while (line := rep_file.readline().rstrip()):
            #  common_entity_list.append(line.split()[1])

    rep_lookup_indices = vocab.lookup_indices(rep_entity_list)
    demo_lookup_indices = vocab.lookup_indices(demo_entity_list)
    common_lookup_indices = vocab.lookup_indices(common_entity_list)

    knowledge_indices['rep'] = rep_lookup_indices
    knowledge_indices['demo'] = demo_lookup_indices
    knowledge_indices['common'] = common_lookup_indices

    print (len(rep_lookup_indices))
    print (len(set(rep_lookup_indices)))

    print (len(demo_lookup_indices))
    print (len(set(demo_lookup_indices)))

    print (len(common_lookup_indices))
    print (len(set(common_lookup_indices)))


    return train_iter, test_iter, vocab, num_class, knowledge_indices





def get_dataloaders(train_iter, test_iter, vocab, batch_size, max_sentence, device):
    """
        Args:
        Returns:
            vocab_size:
            num_class:
    """
    tokenizer = get_tokenizer('basic_english')

    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)

    train_size = int(len(train_dataset) * 1)
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # logging for data statistics
    print('  - Training data size: {}'.format(len(train_dataset)))
    print('  - Validataion data size: {}'.format(len(val_dataset)))
    print('  - Test data size: {}'.format(len(test_dataset)))

    def collate_batch(batch): # split a label and text in each row
        title_pipeline = lambda x: vocab(tokenizer(str(x)))
        text_pipeline = lambda x: vocab(tokenizer(x))
        label_pipeline = lambda x: int(x)

        label_list, title_list, text_list, sentence_list = [], [], [], []
        for (_label, _title, _text) in batch:
            label_list.append(label_pipeline(_label))
            title_indices = title_pipeline(_title)
            text_indices = text_pipeline(_text)

            # pad/trucate each article embedding according to maximum article length
            text_size = len(text_indices)
            title_size = len(title_indices)

            s_list = []
            sentence_tmp = [] 

            for w_idx in text_indices:
                if w_idx == 1: # end of sentence
                    s_list.append(sentence_tmp)
                    sentence_tmp = []
                else:
                    sentence_tmp.append(w_idx)

            sentence_count = 0
            preprocess_sentence_list = []

            for i, sentence in enumerate(s_list):
                if i >= max_sentence:
                    break
                if len(sentence) < MAX_WORDS:
                    for _ in range(MAX_WORDS - len(sentence)):
                        sentence.append(vocab['<unk>'])
                elif len(sentence) > MAX_WORDS:
                    sentence = sentence[:MAX_WORDS]
                else:
                    pass
                preprocess_sentence_list.append(sentence)

            if len(preprocess_sentence_list) < max_sentence:
                for _ in range(max_sentence - len(preprocess_sentence_list)):
                    preprocess_sentence_list.append([0]*MAX_WORDS)

            sentence_list.append(preprocess_sentence_list)

            max_len = max_sentence * MAX_WORDS
            if text_size < max_len:
                padding_size = max_len - text_size
                for _ in range(padding_size):
                    text_indices.append(vocab['<unk>'])
            elif text_size > max_len:
                text_indices = text_indices[:max_len]
            else:
                pass
            
            if title_size < max_len:
                padding_size = max_len - title_size
                for _ in range(padding_size):
                    title_indices.append(vocab['<unk>'])
            elif title_size > max_len:
                title_indices = title_indices[:max_len]
            else:
                pass

            title_list.append(title_indices)
            text_list.append(text_indices) 


        label_list = torch.tensor(label_list, dtype=torch.int64)
        title_list = torch.tensor(title_list, dtype=torch.int64)
        text_list = torch.tensor(text_list, dtype=torch.int64)
        sentence_list = torch.tensor(sentence_list, dtype=torch.int64)
        return label_list.to(device), title_list.to(device), text_list.to(device), sentence_list.to(device)


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_dataloader, valid_dataloader, test_dataloader
