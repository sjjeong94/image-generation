import torch.nn as nn
from torch.nn.utils import spectral_norm

from utils import *


def Generator(dim_x, dim_z):
    # DCGAN Generator
    return nn.Sequential(
        nn.ConvTranspose2d(dim_z, 8*dim_x, 4, 1, 0, bias=False),
        nn.BatchNorm2d(8*dim_x),
        nn.ReLU(True),
        nn.ConvTranspose2d(8*dim_x, 4*dim_x, 4, 2, 1, bias=False),
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
        nn.Conv2d(4*dim_x, 8*dim_x, 4, 2, 1, bias=False),
        nn.BatchNorm2d(8*dim_x),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(8*dim_x, 1, 4, 1, 0, bias=True),
    )


def sn_wrapper(module, sn=False):
    if sn:
        return spectral_norm(module)
    else:
        return module


def G_exp(dim_x, dim_z, sn=False):
    return nn.Sequential(
        sn_wrapper(nn.ConvTranspose2d(
            dim_z, 8*dim_x, 4, 1, 0, bias=False), sn),
        nn.BatchNorm2d(8*dim_x),
        nn.LeakyReLU(0.2, True),
        sn_wrapper(nn.ConvTranspose2d(
            8*dim_x, 4*dim_x, 4, 2, 1, bias=False), sn),
        nn.BatchNorm2d(4*dim_x),
        nn.LeakyReLU(0.2, True),
        sn_wrapper(nn.ConvTranspose2d(
            4*dim_x, 2*dim_x, 4, 2, 1, bias=False), sn),
        nn.BatchNorm2d(2*dim_x),
        nn.LeakyReLU(0.2, True),
        sn_wrapper(nn.ConvTranspose2d(
            2*dim_x, dim_x, 4, 2, 1, bias=False), sn),
        nn.BatchNorm2d(dim_x),
        nn.LeakyReLU(0.2, True),
        sn_wrapper(nn.ConvTranspose2d(dim_x, 3, 4, 2, 1, bias=False), sn),
        nn.Tanh(),
    )


def D_exp(dim_x, sn=False):
    return nn.Sequential(
        sn_wrapper(nn.Conv2d(3, dim_x, 4, 2, 1, bias=False), sn),
        nn.BatchNorm2d(dim_x),
        nn.LeakyReLU(0.2, True),
        sn_wrapper(nn.Conv2d(dim_x, 2*dim_x, 4, 2, 1, bias=False), sn),
        nn.BatchNorm2d(2*dim_x),
        nn.LeakyReLU(0.2, True),
        sn_wrapper(nn.Conv2d(2*dim_x, 4*dim_x, 4, 2, 1, bias=False), sn),
        nn.BatchNorm2d(4*dim_x),
        nn.LeakyReLU(0.2, True),
        sn_wrapper(nn.Conv2d(4*dim_x, 8*dim_x, 4, 2, 1, bias=False), sn),
        nn.BatchNorm2d(8*dim_x),
        nn.LeakyReLU(0.2, True),
        sn_wrapper(nn.Conv2d(8*dim_x, 1, 4, 1, 0, bias=True), sn),
    )


if __name__ == '__main__':

    dim_x = 64
    dim_z = 512

    gen = G_exp(dim_x, dim_z, sn=True)
    dis = D_exp(dim_x, sn=True)
    print(gen)
    print(dis)

    data_root = './data/celeba_hq_64'
    logs_root = './logs/celeba_hq_64'
    # prepare_data('../data/celeba_hq', data_root, 64)
    dataset = get_custum_dataset(data_root)

    train(logs_root, dataset, gen, dis, dim_z, lr_d=0.004, lr_g=0.001)
    test(logs_root, gen, dim_z)
