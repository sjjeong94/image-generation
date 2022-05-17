import torch.nn as nn
from utils import *


def Generator(dim_x, dim_z):
    # DCGAN Generator
    return nn.Sequential(
        nn.ConvTranspose2d(dim_z, 4*dim_x, 4, 1, 0, bias=False),
        nn.BatchNorm2d(4*dim_x),
        nn.ReLU(True),
        nn.ConvTranspose2d(4*dim_x, 2*dim_x, 4, 2, 1, bias=False),
        nn.BatchNorm2d(2*dim_x),
        nn.ReLU(True),
        nn.ConvTranspose2d(2*dim_x, dim_x, 4, 2, 1, bias=False),
        nn.BatchNorm2d(dim_x),
        nn.ReLU(True),
        nn.ConvTranspose2d(dim_x, 3, 4, 2, 1, bias=False),
        nn.Tanh(),
    )


def Discriminator(dim_x):
    # DCGAN Discriminator
    return nn.Sequential(
        nn.Conv2d(3, dim_x, 4, 2, 1, bias=False),
        nn.BatchNorm2d(dim_x),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(dim_x, 2*dim_x, 4, 2, 1, bias=False),
        nn.BatchNorm2d(2*dim_x),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(2*dim_x, 4*dim_x, 4, 2, 1, bias=False),
        nn.BatchNorm2d(4*dim_x),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(4*dim_x, 1, 4, 1, 0, bias=True),
    )


if __name__ == '__main__':

    dim_x = 128
    dim_z = 100

    gen = Generator(dim_x, dim_z)
    dis = Discriminator(dim_x)

    cifar10 = True
    if cifar10:
        logs_root = './logs/cifar10'
        data_root = './data/cifar10'
        dataset = get_cifar10_dataset(data_root)
    else:
        logs_root = './logs/celeba_hq_32'
        data_root = './data/celeba_hq_32'
        # prepare_data('../data/celeba_hq', data_root, 32)
        dataset = get_custum_dataset(data_root)

    train(logs_root, dataset, gen, dis, dim_z)
    test(logs_root, gen, dim_z)
