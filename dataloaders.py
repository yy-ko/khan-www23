import torch, logging, sys
from torch.utils.data import dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset



def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)

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
                label_list.append(label_pipeline(_label))
                processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64) 
                text_list.append(processed_text) 
                offsets.append(processed_text.size(0)) 

            label_list = torch.tensor(label_list, dtype=torch.int64) 
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0) 
            text_list = torch.cat(text_list)

            return label_list.to(device), text_list.to(device), offsets.to(device)

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
        num_class = 5 #  num_class = 3
        tokenizer = get_tokenizer('basic_english')

        train_iter = OUR_DATASET(split='train') # (TODO: to be implemented)

        vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])

        def collate_batch(batch): # split a label and text in each row
            text_pipeline = lambda x: vocab(tokenizer(x))
            label_pipeline = lambda x: int(x) - 1

            label_list, text_list, offsets = [], [], [0] 
            for (_label, _text) in batch: 
                label_list.append(label_pipeline(_label))
                processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64) 
                text_list.append(processed_text) 
                offsets.append(processed_text.size(0)) 

            label_list = torch.tensor(label_list, dtype=torch.int64) 
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0) 
            text_list = torch.cat(text_list)

            return label_list.to(device), text_list.to(device), offsets.to(device)


        train_iter, test_iter = OUR_DATASET() # train, test
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)

        train_size = int(len(train_dataset) * 0.9) # 90:10 train:validation
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        return train_dataloader, valid_dataloader, test_dataloader, len(vocab), num_class

    else:
        logging.error('Invalid dataset name!')
        sys.exit(1)
