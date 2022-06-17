from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))




def get_data():
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    #  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return train_data, val_data, test_data








# ---------------------- Data Loader ---------------------- #
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

    return train_loader, valid_loader


