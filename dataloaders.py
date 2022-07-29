import torch, logging, sys
from torch.utils.data import dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torchtext.legacy import data
# from torchtext.legacy.data import TabularDataset
# from torchtext.legacy.data import Iterator

import pandas as pd
from sklearn.model_selection import train_test_split

def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(str(text))

def yield_tokens_2(data_iter, tokenizer):
    for batch in data_iter:
        print(batch.text)
        yield tokenizer(str(batch.text))

def yield_tokens_r(df, tokenizer):
    for i, row in df.iterrows():
        label = row.label
        text = row.text
        yield tokenizer(str(text))

def preprocess_text (text):
    text = text.str.lower() # lowercase
    text = text.str.replace(r"\#","") # replaces hashtags
    text = text.str.replace(r"http\S+","URL")  # remove URL addresses
    text = text.str.replace(r"@","")
    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    return text

class DataFrameDataset(data.Dataset):
    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.label 
            text = row.text
            examples.append(data.Example.fromlist([text, label], fields))
        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

def get_dataloaders(dataset, data_path, batch_size, device):

    if dataset == 'AGNEWS': # tutorial
        train_iter = AG_NEWS(split='train')
        num_class = len(set([label for (label, text) in train_iter]))
        
        tokenizer = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])

        def collate_batch(batch): # split a label and text in each row
            text_pipeline = lambda x: vocab(tokenizer(x))
            label_pipeline = lambda x: int(x) - 1

            label_list, text_list, offsets = [], [], [0] 
            for (_label, _text) in batch:
                # print(_text, _label)
                label_list.append(label_pipeline(_label))
                processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64) 
                text_list.append(processed_text)
                offsets.append(processed_text.size(0))
            # print(label_list)
            # print(text_list)
            # print(offsets)
            label_list = torch.tensor(label_list, dtype=torch.int64)
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0) 
            text_list = torch.cat(text_list)

            return label_list.to(device), text_list.to(device), offsets.to(device)

        train_iter, test_iter = AG_NEWS() # train, test
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
    
    elif dataset =='ALLSIDES':
        num_class = 5
        tokenizer = get_tokenizer('basic_english')
        data_path = 'dataset/khan_dataset.csv'
        dataset = pd.read_csv(data_path)
        dataset["text"]= preprocess_text(dataset["text"].astype(str))
        dataset = dataset[['text','label']]
        print(dataset.head())
        
        train_df, test_df = train_test_split(dataset, train_size=0.9)
        # print(train_df.values)
        # print(test_df.head())
        v_train = train_df.values
        train_iter = list(map(lambda x: (x.tolist()[1], x.tolist()[0]), v_train))
        v_test = test_df.values
        test_iter = list(map(lambda x: (x.tolist()[1], x.tolist()[0]), v_test))        
        
        vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])

        def collate_batch(batch): # split a label and text in each row
            text_pipeline = lambda x: vocab(tokenizer(x))
            label_pipeline = lambda x: int(x)

            label_list, text_list, offsets = [], [], [0] 
            for (_label, _text) in batch:
                # print(_text, _label)
                label_list.append(label_pipeline(_label))
                processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64) 
                text_list.append(processed_text)
                offsets.append(processed_text.size(0))
            # print(label_list)
            # print(text_list)
            # print(offsets)
            label_list = torch.tensor(label_list, dtype=torch.int64)
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0) 
            text_list = torch.cat(text_list)

            return label_list.to(device), text_list.to(device), offsets.to(device)
        
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
