import torch


def data_loader(data, batch_size, cuda, num_workers):
    args = {
        'shuffle': True,
        'batch_size': batch_size
    }

    if cuda:
        args['num_workers'] = num_workers
        args['pin_memory'] = True

    return torch.utils.data.DataLoader(data, **args)
