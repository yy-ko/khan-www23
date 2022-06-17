from __future__ import print_function

import argparse
import random, os, sys
import numpy as np
import time
import logging
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
#  import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR

import dataloaders, models
from models import TransformerModel

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

bptt = 35


# for Reproducibility
def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data

def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target




def train(model: nn.Module, device) -> None:
    model.train()  # turn on train mode

    train_correct = 0
    total_loss = 0.
    log_interval = 200
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    num_batches = len(train_data) // bptt


    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        batch_size = data.size(0)
        if batch_size != bptt:  # only on last batch 
            src_mask = src_mask[:batch_size, :batch_size]

		# forward and backward passes
        output = model(data, src_mask) 
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad() 
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) 
        optimizer.step()

        total_loss += loss.item() 
        if batch % log_interval == 0 and batch > 0: 
            lr = scheduler.get_last_lr()[0] 
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval 
            cur_loss = total_loss / log_interval 
            ppl = math.exp(cur_loss) 
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
				  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
				  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}') 
            total_loss = 0 
            start_time = time.time()



def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            batch_size = data.size(0)
            if batch_size != bptt:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += batch_size * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)



def main():
    parser = argparse.ArgumentParser(description='Parsing parameters')

    # for multi-gpu
    #  parser.add_argument("--num_workers", type=int, default=1, help="The number of workers per node.")
    #  parser.add_argument("--backend", type=str, default='nccl', help="Backend for Distributed PyTorch: nccl, gloo, mpi")
    #  parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")

    # models & datasets
    parser.add_argument('--model', type=str, default='RESNET18', help='Name of Model.')
    parser.add_argument("--method", type=str, default='LSW', help="LR scaling method for large batch training.")
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Name of dataset.')
    parser.add_argument('--data_path', type=str, default='/data', help='Data path.')

    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--model_dir', type=str, default='../trained_models', help='Path for saving the trained model')

    # user-defined parameters
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size for one process.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate.")
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
    logging.info('===============================TRAIN INFO START===============================')
    logging.info('  - TRAINING MODEL = %s' % (args.model))
    logging.info('  - DATASET = %s' % (args.dataset))
    logging.info('  - BATCH SIZE = ' + str(args.batch_size))
    logging.info('  - NUM EPOCHS = ' + str(args.num_epochs))
    logging.info('  - LEARNING RATE = ' + str(args.learning_rate))
    # add more if needed
    logging.info('=============================== TRAIN INFO END ===============================')



    # ------------------------------------------------------------------------#
    # ---------------------------- Training Setup ----------------------------#
    # ------------------------------------------------------------------------#

    # Get the model and data loaders
    #  device = torch.device("cuda:{}".format(local_rank))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# (TODO) model design
	ntokens = len(vocab)  # size of vocabulary
	emsize = 200  # embedding dimension
	d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
	nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
	nhead = 2  # number of heads in nn.MultiheadAttention
	dropout = 0.2  # dropout probability


	# preparing the model and data
	model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)
    model = model.to(device)

	train_data, val_data, test_data = dataloaders.get_data()
	train_data = batchify(train_data, args.batch_size)  # shape [seq_len, batch_size]
	val_data = batchify(val_data, args.batch_size)
	test_data = batchify(test_data, args.batch_size)

    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate) # optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95) # learning rate scheduling

    start_time = time.time()
    total_start_time = start_time
    max_accuracy = 0

    num_batches = 0
    num_updates = 0


    # ----------------------------------------------------------------------------#
    # ---------------------------- Main Training Loop ----------------------------#
    # ----------------------------------------------------------------------------#

    logging.info('')
    logging.info('=============================== Training Start ===============================')
    logging.info('\tepoch\tstep\ttrain\ttest\tloss\tthroughput')


    for epoch in range(args.num_epochs):
        epoch_start_time = time.time() 
        train(model, train_data) 
        val_loss = evaluate(model, val_data)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
			  f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_model = copy.deepcopy(model)

		scheduler.step()


        elapsed_time = time.time() - start_time
        # ---------------------------- end of each training epoch ---------------------------- #


        # --------------------------------- EVALUATION --------------------------------- #
        # ----- at the end of every epoch, evaluating the current model a ccuracy ------ #
        # ------------------------------------------------------------------------------ #
        #  train_accuracy = train_correct / len(train_loader.dataset) * world_size
        #  test_accuracy = evaluate(model=model, device=device, test_loader=test_loader)

        #  if test_accuracy > max_accuracy:
            #  max_accuracy = test_accuracy

        #  throughput = len(train_loader.dataset) / elapsed_time
        #  current_lr = optimizer.param_groups[0]['lr']

        #  logging.info('\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}'.format(
            #  (epoch+1),
            #  num_updates,
            #  train_accuracy,
            #  test_accuracy,
            #  loss.item(),
            #  throughput)
            #  )

        #  start_time = time.time()

        # ------------------------- Save Point ------------------------- #
        #  if args.save_model:
            #  torch.save(model.state_dict(), args.model_dir)



    # ----------------------------------------------------------------------------#
    # ----------------------------- End of Training ------------------------------#
    # ----------------------------------------------------------------------------#
    total_train_time = time.time() - total_start_time
    logging.info('')
    logging.info('=============================== Training End ===============================')
    logging.info('Final Test Accuracy: {:.4f}, Total training time: {:.2f} (sec.)'.format(max_accuracy, total_train_time))
    logging.info('============================================================================')



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()



