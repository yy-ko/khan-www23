from __future__ import print_function

import torch
import torch.distributed as dist

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler



def get_dataset(dataset, data_path, batch_size, world_size):
    # Prepare dataset and dataloader
    if dataset == 'MNIST':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

        train_set = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    elif dataset == 'CIFAR10':
        #  weight_decay = 5e-4
        pin_memory=False

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        transform_validation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_set = datasets.CIFAR10(data_path, train=True, download=False, transform=transform_train) 
        valid_set = datasets.CIFAR10(data_path, train=True, download=False, transform=transform_validation)
        test_set = datasets.CIFAR10(data_path, train=False, download=False, transform=transform_test)


        val_size = 5000 # 10% of training samples
        train_size = len(train_set) - val_size
        train_set, _ = random_split(train_set, [train_size, val_size])
        _, valid_set = random_split(valid_set, [train_size, val_size])


    elif dataset == 'CIFAR100':
        #  weight_decay = 5e-4

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            #  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_validation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            #  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = datasets.CIFAR100(data_path, train=True, download=False, transform=transform_train) 
        valid_set = datasets.CIFAR100(data_path, train=True, download=False, transform=transform_validation)


        val_size = 5000 # 10% of training samples
        train_size = len(train_set) - val_size
        train_set, _ = random_split(train_set, [train_size, val_size])
        _, valid_set = random_split(valid_set, [train_size, val_size])

    elif dataset == 'IMAGENET':
        #  weight_decay = 1e-4
        pin_memory=True

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_validation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_set = datasets.ImageFolder("/data/imageNet/raw-data/train", transform=transform_train) 
        valid_set = datasets.ImageFolder("/data/imageNet/raw-data/validation", transform=transform_validation)



    # Restricts data loading to a subset of the dataset exclusive to the current process
    dist_sampler = DistributedSampler(dataset=train_set, num_replicas=world_size)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=dist_sampler, num_workers=4)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False, num_workers=4)

    #  train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=dist_sampler, num_workers=4, pin_memory=pin_memory)
    #  valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)
    #  test_loader = DataLoader(dataset=test_set, batch_size=200, shuffle=False, num_workers=4)

    return train_loader, valid_loader



# add user-defined dataloaders

