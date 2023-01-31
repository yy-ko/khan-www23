from __future__ import print_function

import argparse
import random
import numpy as np
import time
import logging
import warnings
from tqdm import tqdm
import torch
from torch import nn
from scipy.special import softmax
from torch.optim.lr_scheduler import StepLR

import data_utils
from models import KHANModel
import gc
gc.collect()
torch.cuda.empty_cache()

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

# for Reproducibility
def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    #  pass

parser = argparse.ArgumentParser(description='Parsing parameters')
parser.add_argument("--gpu_index", type=int, help="GPU index. Necessary for specifying the CUDA device.")
# models & datasets
parser.add_argument('--model', type=str, default='KHAN', help='Name of Model.')
parser.add_argument("--embed_size", type=int, default=128, help="Word/Sentence Embedding size.")
parser.add_argument('--max_sentence', type=int, default=40, help='Maximum Number of Sentences in each document.')
parser.add_argument('--num_layer', type=int, default=1, help='Number of Transformer Encoder Layers.')
parser.add_argument('--num_head', type=int, default=4, help='Number of Multihead Attentions.')
parser.add_argument('--d_hid', type=int, default=256, help='Dimension of a hidden layer.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability.')
parser.add_argument('--alpha', type=float, default=0.5, help='Weight of common knowledge.')
parser.add_argument('--beta', type=float, default=0.5, help='Weight of political knowledge.')

parser.add_argument('--dataset', type=str, default='SEMEVAL', help='Name of dataset (SEMEVAL, ALLSIDES).')
parser.add_argument('--data_path', type=str, default='./data', help='Data path.')
parser.add_argument('--save_model', action='store_false', default=False, help='For Saving the current Model')
parser.add_argument('--model_dir', type=str, default='../trained_models', help='Path for saving the trained model')

# user-defined hyperparameters
parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation and test batch size.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)') # for reproducibility

args = parser.parse_args()

# Get the dataloaders and model
device = torch.device("cuda:{}".format(args.gpu_index))
print('Count of using GPUs:', torch.cuda.device_count())

def evaluate(model, device, dataloader) -> float:
    model.eval() # turn on eval mode
    total_acc = 0
    total_count = 0
    with torch.no_grad():
        for idx, (labels, titles, sentences) in enumerate(dataloader):
            predicted_label = model(sentences, titles)
            total_acc += (predicted_label.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    return total_acc/total_count

def train_each_fold(train_iter, test_iter, vocab, num_class, knowledge_list, fold_idx, k_folds, sampler):
    
    train_iter, test_iter, vocab, num_class, knowledge_list, fold_idx, k_folds, sampler = train_iter, test_iter, vocab, num_class, knowledge_list, fold_idx, k_folds, sampler

    train_data, val_data, test_data = data_utils.get_dataloaders(train_iter, test_iter, vocab, args.batch_size, args.max_sentence, sampler, device)

    nhead = args.num_head
    d_hid = args.d_hid
    dropout = args.dropout
    nlayers = args.num_layer
    alpha = args.alpha
    beta = args.beta

    model = KHANModel(len(vocab), args.embed_size, nhead, d_hid, nlayers, dropout, num_class, knowledge_list, alpha, beta)
    params = list(model.parameters())
    print(len(params))
    print(params[0].size())
    
    model = model.to(device) # model to GPU
    total = sum([param.nelement() for param in model.parameters()])
    print("Total params: %.2fM" % (total/1e6))

    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05) # optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=1) # learning rate scheduling
    
    start_time = time.time()
    total_start_time = start_time
    max_accuracy = 0
    num_batches = 0
    
    for epoch in range(args.num_epochs): 
        epoch_start_time = time.time() 
        model.train()  # turn on train mode 
        train_correct, train_count = 0, 0 
        start_time = time.time()
        total_loss = 0.

        # ------------------------ Epoch Start ------------------------ #
        for idx, (labels, titles, sentences) in enumerate(train_data):
            optimizer.zero_grad()
            predicted_labels = model(sentences, titles)
            loss = criterion(predicted_labels, labels)
            loss.backward()
            
            # Gradient Clipping
            max_grad_norm = 3
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_correct += (predicted_labels.argmax(1) == labels).sum().item()
            train_count += labels.size(0) 
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_data)
        scheduler.step(epoch_loss)
        epoch_time = time.time() - epoch_start_time
        # ------------------------ Epoch End ------------------------ #

        # --------------------------------- EVALUATION --------------------------------- #
        # ----- at the end of every epoch, evaluating the current model a ccuracy ------ #
        # ------------------------------------------------------------------------------ #
        train_accuracy = train_correct / train_count
        val_accuracy = evaluate(model, device, test_data)

        print('Fold: {:3d} | Epoch: {:3d} | Loss: {:6.4f} | TrainAcc: {:6.4f} | ValAcc: {:6.4f} | Time: {:5.2f}'.format(
            fold_idx,
            (epoch+1),
            total_loss/ len(train_data), 
            train_accuracy,
            val_accuracy,
            epoch_time)
            )

        # ------------------------- Save Best Model ------------------------- #
        if val_accuracy > max_accuracy:
            max_accuracy = val_accuracy

        if args.save_model:
            torch.save(model.state_dict(), args.model_dir)

    # ----------------------------------------------------------------------------#
    # ----------------------------- End of Training ------------------------------#
    # ----------------------------------------------------------------------------#

    fold_train_time = time.time() - total_start_time

    print('')
    print('================================== FOLD - {:3d} =================================='.format(fold_idx))
    print('============= Test Accuracy: {:.4f}, Training time: {:.2f} (sec.) ================'.format(max_accuracy, fold_train_time))
    print('================================================================================')
    
    return max_accuracy, fold_train_time


def main():

    # ----------------------------------------------------------------------------#
    # ---------------------------- Main Training Loop ----------------------------#
    # ----------------------------------------------------------------------------#
    print('')
    print('==================================== Training Start ====================================')

    data_utils.train_datasets(args.dataset, args.data_path)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Set random seeds for reproducibility
    set_random_seeds(args.seed)

    # Summary of training information
    print('====================================TRAIN INFO START====================================')
    print('  - TRAINING MODEL = %s' % (args.model))
    print('     - Embedding Size = %s' % (args.embed_size))
    print('     - Maximum Length = %s' % (args.max_sentence))
    print('     - Number of Transformer Encoder Layers = %s' % (args.num_layer))
    print('     - Number of Multi-head Attentions = %s' % (args.num_head))
    print('     - Hidden Layer Dimension = %s' % (args.d_hid))
    print('     - Dropout Probability = %s' % (args.dropout))
    print('     - Alpha = %s' % (args.alpha))
    print('     - Beta = %s' % (args.beta))
    print('  - DATASET = %s' % (args.dataset))
    print('  - BATCH SIZE = ' + str(args.batch_size))
    print('  - NUM EPOCHS = ' + str(args.num_epochs))
    print('  - LEARNING RATE = ' + str(args.learning_rate))
    print('   ')
    # add more if needed
    
    main()
