import torch, logging, sys
from torch.utils.data import dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset


from collections import Counter, OrderedDict
from typing import Iterable
import pandas as pd
from sklearn.model_selection import train_test_split


def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(str(text))

def yield_tokens_2(data_iter, tokenizer):
    for _, title, text in data_iter:
        yield tokenizer(str(text))
        yield tokenizer(str(title))

def preprocess_text (text):
    text = text.str.lower() # lowercase
    text = text.str.replace(r"\#","") # replaces hashtags
    text = text.str.replace(r"http\S+","URL")  # remove URL addresses
    text = text.str.replace(r"@","")
    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    return text


def build_knowledge_index_from_iterator(iterator: Iterable):
    """
    Build knowledge index from an iterator.

    Args:
        iterator:

    Returns:
    """
    counter = Counter()
    for tokens in iterator: 
        counter.update(tokens)

    #  specials = specials or []

    # First sort by descending frequency, then lexicographically
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0])) 
    #  if max_tokens is None:
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    print (ordered_dict)

    return None




def get_dataloaders(dataset, data_path, batch_size, max_len, device):

    if dataset == 'AGNEWS': # tutorial
        num_class = 4
        train_iter = AG_NEWS(split='train')

        tokenizer = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])

        train_iter = AG_NEWS(split='train')
        knowledge = build_knowledge_index_from_iterator(yield_tokens(train_iter, tokenizer))

        def collate_batch(batch): # split a label and text in each row
            text_pipeline = lambda x: vocab(tokenizer(x))
            label_pipeline = lambda x: int(x) - 1

            label_list, text_list = [], []
            for (_label, _text) in batch: 
                label_list.append(label_pipeline(_label))

                text_indices = text_pipeline(_text)
                # pad/trucate each article embedding according to maximum article length
                text_size = len(text_indices) 
                if text_size < max_len:
                    padding_size = max_len - text_size
                    for _ in range(padding_size):
                        text_indices.append(vocab['unk'])
                elif text_size > max_len:
                    text_indices = text_indices[:max_len]
                else:
                    pass

                text_list.append(text_indices) 

            label_list = torch.tensor(label_list, dtype=torch.int64) 
            text_list = torch.tensor(text_list, dtype=torch.int64)

            return label_list.to(device), text_list.to(device)
            #  return label_list.to(device), text_list.to(device), offsets.to(device)

        train_iter, test_iter = AG_NEWS() # train, test
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)

        train_size = int(len(train_dataset) * 0.9)
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        return train_dataloader, valid_dataloader, test_dataloader, len(vocab), num_class


    elif dataset =='ALLSIDES':
        num_class = 5
        tokenizer = get_tokenizer('basic_english')
        data_path = 'data/khan_dataset.csv'
        # data_path = 'data/news_dataset_test.csv'
        dataset = pd.read_csv(data_path)
        dataset["text"]= preprocess_text(dataset["text"].astype(str))
        dataset = dataset[['text','label']]
        #  print(dataset.head())

        train_df, test_df = train_test_split(dataset, train_size=0.9)
        v_train = train_df.values
        train_iter = list(map(lambda x: (x.tolist()[1], x.tolist()[0]), v_train))
        v_test = test_df.values
        test_iter = list(map(lambda x: (x.tolist()[1], x.tolist()[0]), v_test))
        vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])

        def collate_batch(batch): # split a label and text in each row
            text_pipeline = lambda x: vocab(tokenizer(x))
            label_pipeline = lambda x: int(x)

            label_list, text_list = [], []
            for (_label, _text) in batch:
                label_list.append(label_pipeline(_label))
                text_indices = text_pipeline(_text)
                # pad/trucate each article embedding according to maximum article length
                text_size = len(text_indices) 
                if text_size < max_len:
                    padding_size = max_len - text_size
                    for _ in range(padding_size):
                        text_indices.append(vocab['unk'])
                elif text_size > max_len:
                    text_indices = text_indices[:max_len]
                else:
                    pass

                text_list.append(text_indices) 

            label_list = torch.tensor(label_list, dtype=torch.int64) 
            text_list = torch.tensor(text_list, dtype=torch.int64)
            return label_list.to(device), text_list.to(device)

        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)
        #  print('train data len : {}'.format(len(train_dataset)))
        #  print('test data len : {}'.format(len(test_dataset)))

        train_size = int(len(train_dataset) * 0.9)
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        return train_dataloader, valid_dataloader, test_dataloader, len(vocab), num_class

    elif dataset =='KCD':
        num_class = 2
        tokenizer = get_tokenizer('basic_english')
        # data_path = 'data/khan_dataset.csv'
        data_path = 'data/kcd_dataset.csv'
        dataset = pd.read_csv(data_path)
        dataset["text"]= preprocess_text(dataset["text"].astype(str))
        dataset = dataset[['text','label']]
        print(dataset.head())
        
        train_df, test_df = train_test_split(dataset, train_size=0.9)
        v_train = train_df.values
        train_iter = list(map(lambda x: (x.tolist()[1], x.tolist()[0]), v_train))
        v_test = test_df.values
        test_iter = list(map(lambda x: (x.tolist()[1], x.tolist()[0]), v_test))        
        
        vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])

        def collate_batch(batch): # split a label and text in each row
            text_pipeline = lambda x: vocab(tokenizer(x))
            label_pipeline = lambda x: int(x)

            label_list, text_list = [], []
            for (_label, _text) in batch:
                label_list.append(label_pipeline(_label))
                
                text_indices = text_pipeline(_text)
                # pad/trucate each article embedding according to maximum article length
                text_size = len(text_indices) 
                if text_size < max_len:
                    padding_size = max_len - text_size
                    for _ in range(padding_size):
                        text_indices.append(vocab['unk'])
                elif text_size > max_len:
                    text_indices = text_indices[:max_len]
                else:
                    pass

                text_list.append(text_indices) 
            label_list = torch.tensor(label_list, dtype=torch.int64) 
            text_list = torch.tensor(text_list, dtype=torch.int64)
                        
            return label_list.to(device), text_list.to(device)
        
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)
        print('train data len : {}'.format(len(train_dataset)))
        print('test data len : {}'.format(len(test_dataset)))

        train_size = int(len(train_dataset) * 0.9)
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        return train_dataloader, valid_dataloader, test_dataloader, len(vocab), num_class

    # title 포함 (잠시 중단)
    elif dataset =='ALLSIDES_2':
        num_class = 5
        tokenizer = get_tokenizer('basic_english')
        # data_path = 'data/khan_dataset.csv'
        data_path = 'data/news_dataset_test.csv'
        dataset = pd.read_csv(data_path)
        dataset["text"]= preprocess_text(dataset["text"].astype(str))
        dataset = dataset[['title','text','label']]
        print(dataset.head())
        
        train_df, test_df = train_test_split(dataset, train_size=0.9)
        v_train = train_df.values
        train_iter = list(map(lambda x: (x.tolist()[2], x.tolist()[0], x.tolist()[1]), v_train))
        v_test = test_df.values
        test_iter = list(map(lambda x: (x.tolist()[2], x.tolist()[0], x.tolist()[1]), v_test))        
        
        vocab = build_vocab_from_iterator(yield_tokens_2(train_iter, tokenizer), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])

        def collate_batch(batch): # split a label and text in each row
            title_pipeline = lambda x: vocab(tokenizer(x))
            text_pipeline = lambda x: vocab(tokenizer(x))
            label_pipeline = lambda x: int(x)

            label_list, title_list, text_list = [], [], []
            for (_label, _title, _text) in batch:
                label_list.append(label_pipeline(_label))
                
                text_indices = text_pipeline(_text)
                title_indices = title_pipeline(_title)
                # pad/trucate each article embedding according to maximum article length
                text_size = len(text_indices)
                title_size = len(title_indices)
                if text_size < max_len:
                    padding_size = max_len - text_size
                    for _ in range(padding_size):
                        text_indices.append(vocab['unk'])
                elif text_size > max_len:
                    text_indices = text_indices[:max_len]
                else:
                    pass

                text_list.append(text_indices) 
            # print(label_list)
            # print(text_list)
            label_list = torch.tensor(label_list, dtype=torch.int64) 
            text_list = torch.tensor(text_list, dtype=torch.int64)
                        
            return label_list.to(device), text_list.to(device)
        
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)
        print('train data len : {}'.format(len(train_dataset)))
        print('test data len : {}'.format(len(test_dataset)))

        train_size = int(len(train_dataset) * 0.9)
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        return train_dataloader, valid_dataloader, test_dataloader, len(vocab), num_class

    else:
        logging.error('Invalid dataset name!')
        sys.exit(1)
