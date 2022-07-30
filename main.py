from __future__ import print_function

import argparse
import random, os, sys
import numpy as np
import time
import logging
import warnings
from tqdm import tqdm

import math
from typing import Tuple

import torch
from torch import nn, Tensor
from torch import optim

#  import torch.distributed as dist
#  from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim.lr_scheduler import StepLR

import dataloaders, models
from models import KHANModel

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

def evaluate(model, device, dataloader) -> float:
    model.eval()
    total_acc = 0 # here?
    total_count = 0

    with torch.no_grad():
        for idx, (labels, texts) in enumerate(dataloader):
            predicted_label = model(texts)
            total_acc += (predicted_label.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    return total_acc/total_count



def main():
    parser = argparse.ArgumentParser(description='Parsing parameters')

    # for multi-gpu
    #  parser.add_argument("--num_gpus", type=int, default=1, help="The number of GPUs.")
    #  parser.add_argument("--backend", type=str, default='nccl', help="Backend for Distributed PyTorch: nccl, gloo, mpi")
    #  parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--gpu_index", type=int, help="GPU index. Necessary for specifying the CUDA device.")

    # models & datasets
    parser.add_argument('--model', type=str, default='KHAN', help='Name of Model.')
    parser.add_argument('--dataset', type=str, default='SEMEVAL', help='Name of dataset.')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum length of each document.')
    parser.add_argument('--data_path', type=str, default='./data', help='Data path.')

    parser.add_argument('--save_model', action='store_false', default=False, help='For Saving the current Model')
    parser.add_argument('--model_dir', type=str, default='../trained_models', help='Path for saving the trained model')

    # user-defined hyperparameters
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation and test batch size.")
    parser.add_argument("--learning_rate", type=float, default=5, help="Learning rate.")
    parser.add_argument("--embed_size", type=int, default=64, help="Word/Sentennce Embedding size.")
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)') # for reproducibility

    args = parser.parse_args()

    #  dist.init_process_group(backend='nccl')
    #  world_size = dist.get_world_size()
    #  rank = dist.get_rank()

    # Set random seeds for reproducibility
    set_random_seeds(args.seed)

    # Summary of training information
    #  if global_rank == 0:
    print('===============================TRAIN INFO START===============================')
    print('  - TRAINING MODEL = %s' % (args.model))
    print('  - DATASET = %s' % (args.dataset))
    print('  - TRAINING BATCH SIZE = ' + str(args.batch_size))
    print('  - EVAL/TEST BATCH SIZE = ' + str(args.eval_batch_size))
    print('  - NUM EPOCHS = ' + str(args.num_epochs))
    print('  - LEARNING RATE = ' + str(args.learning_rate))
    print('   ')
    # add more if needed
    #  print('==============================================================================')


    # ------------------------------------------------------------------------#
    # ---------------------------- Training Setup ----------------------------#
    # ------------------------------------------------------------------------#
    # Get the dataloaders and model
    #  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda:{}".format(args.gpu_index)) 

    nhead = 4 # 8
    d_hid = 512 # 2048
    dropout = 0.1 # 0.1
    nlayers = 4 # 
    train_data, val_data, test_data, vocab_size, num_class = dataloaders.get_dataloaders(args.dataset, args.data_path, args.batch_size, args.eval_batch_size, args.max_len, device)

    model = KHANModel(vocab_size, args.embed_size, nhead, d_hid, nlayers, dropout, num_class)
    model = model.to(device) # model to GPU

    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9) # optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # learning rate scheduling


    # ----------------------------------------------------------------------------#
    # ---------------------------- Main Training Loop ----------------------------#
    # ----------------------------------------------------------------------------#
    print('')
    print('=============================== Training Start ===============================')



    start_time = time.time()
    total_start_time = start_time
    max_accuracy = 0
    num_batches = 0

    #  for epoch in tqdm(range(args.num_epochs)): 
    for epoch in range(args.num_epochs): 
        epoch_start_time = time.time() 
        model.train()  # turn on train mode 
        train_correct, train_count = 0, 0 
        start_time = time.time()
        total_loss = 0.

        # ------------------------ Epoch Start ------------------------ #
        for idx, (labels, texts) in enumerate(train_data):
            optimizer.zero_grad()
            predicted_labels = model(texts)
            loss = criterion(predicted_labels, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            train_correct += (predicted_labels.argmax(1) == labels).sum().item()
            train_count += labels.size(0) 
            total_loss += loss.item()

        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        # ------------------------ Epoch End ------------------------ #

        # --------------------------------- EVALUATION --------------------------------- #
        # ----- at the end of every epoch, evaluating the current model a ccuracy ------ #
        # ------------------------------------------------------------------------------ #
        train_accuracy = train_correct / train_count
        val_accuracy = evaluate(model, device, test_data)
        #  val_accuracy = evaluate(model, device, val_data)

        print('Epoch: {:3d} | Loss: {:6.4f} | TrainAcc: {:6.4f} | ValAcc: {:6.4f} | Time: {:5.2f}'.format(
            (epoch+1),
            total_loss/ len(train_data), 
            train_accuracy,
            val_accuracy,
            epoch_time)
            )

        # ------------------------- Save Best Model ------------------------- #
        if val_accuracy > max_accuracy:
            max_accuracy = val_accuracy
            #  best_model = copy.deepcopy(model)

        if args.save_model:
            torch.save(model.state_dict(), args.model_dir)

    # (TODO) Measure the final test accuracy
    # ----------------------------------------------------------------------------#
    # ----------------------------- End of Training ------------------------------#
    # ----------------------------------------------------------------------------#
    total_train_time = time.time() - total_start_time
    print('')
    print('=============================== Training End ===============================')
    print('Final Test Accuracy: {:.4f}, Total training time: {:.2f} (sec.)'.format(max_accuracy, total_train_time))
    print('============================================================================')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
