import torch


def mnist():
    # exchange with the corrupted mnist dataset
    train = torch.randn(40000, 784)
    test = torch.randn(5000, 784) 
    #change
    return train, test
