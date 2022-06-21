from __future__ import print_function

import argparse
import random, os, sys
import numpy as np
import time
import logging
import warnings

import math
from typing import Tuple

import torch
from torch import nn, Tensor
from torch import optim 
#  import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR

import dataloaders, models
from models import KhanModel

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
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count



def main():
    parser = argparse.ArgumentParser(description='Parsing parameters')

    # for multi-gpu
    #  parser.add_argument("--num_workers", type=int, default=1, help="The number of workers per node.")
    #  parser.add_argument("--backend", type=str, default='nccl', help="Backend for Distributed PyTorch: nccl, gloo, mpi")
    #  parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")

    # models & datasets
    parser.add_argument('--model', type=str, default='KHAN', help='Name of Model.')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Name of dataset.')
    parser.add_argument('--data_path', type=str, default='/data', help='Data path.')

    parser.add_argument('--save_model', action='store_false', default=False, help='For Saving the current Model')
    parser.add_argument('--model_dir', type=str, default='../trained_models', help='Path for saving the trained model')

    # user-defined hyperparameters
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=100, help="Evaluation and test batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--embed_size", type=int, default=64, help="Word/Sentennce Embedding size.")
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)') # for reproducibility

    args = parser.parse_args()

    #  dist.init_process_group(backend=args.backend)
    #  world_size = dist.get_world_size()
    #  local_rank = args.local_rank
    #  global_rank = dist.get_rank()


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
    # add more if needed



    # ------------------------------------------------------------------------#
    # ---------------------------- Training Setup ----------------------------#
    # ------------------------------------------------------------------------#
    # Get the dataloaders and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #  device = torch.device("cuda:{}".format(local_rank))

    train_data, val_data, test_data, vocab_size, num_class = dataloaders.get_dataloaders(args.dataset, args.data_path, args.batch_size, device)

    model = KhanModel(vocab_size, args.embed_size, num_class)
    model = model.to(device) # model to GPU

    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate) # optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1) # learning rate scheduling



    # ----------------------------------------------------------------------------#
    # ---------------------------- Main Training Loop ----------------------------#
    # ----------------------------------------------------------------------------#
    print('')
    print('=============================== Training Start ===============================')

    start_time = time.time()
    total_start_time = start_time
    max_accuracy = 0
    num_batches = 0

    for epoch in range(args.num_epochs): 
        epoch_start_time = time.time() 
        model.train()  # turn on train mode 
        train_correct, train_count = 0, 0 
        start_time = time.time()

        # ------------------------ Epoch Start ------------------------ #
        for idx, (label, text, offsets) in enumerate(train_data):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            train_correct += (predicted_label.argmax(1) == label).sum().item()
            train_count += label.size(0)

        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        # ------------------------ Epoch End ------------------------ #

        # --------------------------------- EVALUATION --------------------------------- #
        # ----- at the end of every epoch, evaluating the current model a ccuracy ------ #
        # ------------------------------------------------------------------------------ #
        train_accuracy = train_correct / train_count
        val_accuracy = evaluate(model, device, val_data)

        print('Epoch: {:3d} | Loss: {:6.4f} | TrainAcc: {:6.4f} | ValAcc: {:6.4f} | Time: {:5.2f}'.format(
            (epoch+1),
            loss.item(),
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

    # (TODO) Measure the final est accuracy

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



