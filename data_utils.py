import torch, logging, sys
from torch.utils.data import DataLoader, WeightedRandomSampler

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import main

MAX_WORDS = 40

# tokenize title and text
def yield_tokens(data_iter, tokenizer):
    for _, title, text in data_iter:
        yield tokenizer(title)
        yield tokenizer(str(text))

# preprocess article text
def preprocess_text(text):
    text = text.str.lower() # lowercase
    text = text.str.replace(r"\#","") # replaces hashtags
    text = text.str.replace(r"http\S+","")  # remove URL addresses
    return text

# read datasets and train each k-fold
def train_datasets(dataset_n, data_path):
    """
        Args:
            dataset_n:
            data_path:
    """

    num_class = 0
    if dataset_n =='SEMEVAL':
        num_class = 2
        k_folds = 10
        tokenizer = get_tokenizer('basic_english')
        data_path += '/SemEval.csv'

    elif dataset_n =='ALLSIDES-S':
        num_class = 3
        k_folds = 10
        data_path += '/AllSides-S.csv'

    elif dataset_n =='ALLSIDES-L':
        num_class = 5
        k_folds = 10
        data_path += '/AllSides-L.csv'

    else:
        logging.error('Invalid dataset name!')
        sys.exit(1)

    # read a dataset file from a local path
    dataset = pd.read_csv(data_path)
    dataset.dropna(inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    # separate dataset by x, y
    dataset_x = dataset[['text','title']]
    dataset_y = dataset[['label']]
    
    # exploit common-sense and political knowledge
    knowledge_indices = {}
    rep_entity_list = []
    demo_entity_list = []
    common_entity_list = []

    with open('./pre-trained/entities_con.dict') as rep_file:
        while (line := rep_file.readline().rstrip()):
            rep_entity_list.append(line.split()[1])

    with open('./pre-trained/entities_lib.dict') as demo_file:
        while (line := demo_file.readline().rstrip()):
            demo_entity_list.append(line.split()[1])

    with open('./pre-trained/entities_yago.dict') as common_file:
        while (line := common_file.readline().rstrip()):
            common_entity_list.append(line.split()[1].split('_')[0].lower())
    
    # stratified k-fold training
    Skfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=10)
    fold_idx = 0
    total_accuracy = 0
    best_accuracy = 0
    total_train_time = 0
    acc_list = []
    
    # K-th iteration
    # Extraction each fold train/testset and train
    for train_index, test_index in Skfold.split(dataset_x, dataset_y):
        fold_idx += 1
        x_train_df, x_test_df = dataset_x.loc[train_index], dataset_x.loc[test_index]
        y_train_df, y_test_df = dataset_y.loc[train_index], dataset_y.loc[test_index]
        
        # logging for data statistics
        print('  - Training data size: {}'.format(len(y_train_df)))
        print('  - Test data size: {}'.format(len(y_test_df)))
    
        x_train = x_train_df.values
        y_train = y_train_df.values
        x_test = x_test_df.values
        y_test = y_test_df.values
        
        # Weighted Random Sampler
        # class 0 : 366, class 1 : 214
        class_counts = y_train_df.value_counts().to_list() # [366, 214]
        num_samples = sum(class_counts) # 580
        labels = y_train_df.values
        
        # each class weight initialization [580/366, 580/214]
        class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))] 
        print(class_weights)
        
        weights = [class_weights[int(labels[i][0])] for i in range(int(num_samples))]
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

        train_iter = list(map(lambda x, y: (y.tolist()[0], x.tolist()[1], x.tolist()[0]), x_train, y_train ))
        test_iter = list(map(lambda x, y: (y.tolist()[0], x.tolist()[1], x.tolist()[0]), x_test, y_test ))

        # build vocab
        tokenizer = get_tokenizer('basic_english')
        vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=['<unk>', '<sep>'])
        vocab.set_default_index(vocab['<unk>'])
        
        rep_lookup_indices = vocab.lookup_indices(rep_entity_list)
        demo_lookup_indices = vocab.lookup_indices(demo_entity_list)
        common_lookup_indices = vocab.lookup_indices(common_entity_list)

        knowledge_indices['rep'] = rep_lookup_indices
        knowledge_indices['demo'] = demo_lookup_indices
        knowledge_indices['common'] = common_lookup_indices
        
        # train each k-fold
        fold_accuracy, fold_train_time = main.train_each_fold(train_iter, test_iter, vocab, num_class, knowledge_indices, fold_idx, k_folds, sampler)
        total_accuracy += fold_accuracy
        if fold_accuracy > best_accuracy:
            best_accuracy = fold_accuracy
        
        total_train_time += fold_train_time
        acc_list.append(fold_accuracy)
        
        if dataset_n == 'ALLSIDES-S' and fold_idx == 3:
            break
        if dataset_n == 'ALLSIDES-L':
            break
    
    # K-folds Training Result
    print('')
    print('=============================== {:2d}-Folds Training Result ==============================='.format(fold_idx))
    print('=============== Total Accuracy: {:.4f},    Training time: {:.2f} (sec.)   ================'.format(total_accuracy/fold_idx, total_train_time))
    print('=============== Best Accuracy: {:.4f},     Accuracy variance: {:.4f}      ================'.format(best_accuracy, np.var(acc_list)))
    print('========================================================================================')
    print('Accuracy_list: ', acc_list)


def get_dataloaders(train_iter, test_iter, vocab, batch_size, max_sentence, sampler, device):
    """
        Args:
        Returns:
            vocab_size:
            num_class:
    """
    tokenizer = get_tokenizer('basic_english')

    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)

    def collate_batch(batch): # split a label and text in each row
        title_pipeline = lambda x: vocab(tokenizer(str(x)))
        text_pipeline = lambda x: vocab(tokenizer(x))
        label_pipeline = lambda x: int(x)

        label_list, title_list, sentence_list = [], [], []
        for (_label, _title, _text) in batch:
            label_list.append(label_pipeline(_label))
            title_indices = title_pipeline(_title)
            text_indices = text_pipeline(_text)

            # pad/trucate each article embedding according to maximum article length
            text_size = len(text_indices)

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


            title_len = len(title_indices)
            if title_len < MAX_WORDS:
                for _ in range(MAX_WORDS - title_len):
                    title_indices.append(vocab['<unk>'])
            elif title_len > MAX_WORDS:
                title_indices = title_indices[:MAX_WORDS]
            else:
                pass

            title_list.append(title_indices)


        label_list = torch.tensor(label_list, dtype=torch.int64)
        title_list = torch.tensor(title_list, dtype=torch.int64)
        sentence_list = torch.tensor(sentence_list, dtype=torch.int64)
        return label_list.to(device), title_list.to(device), sentence_list.to(device)

    # sampler addition
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler = sampler, collate_fn=collate_batch)
    valid_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_dataloader, valid_dataloader, test_dataloader
