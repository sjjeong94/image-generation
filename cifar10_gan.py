import os
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import albumentations
import albumentations.pytorch
import imageio


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class CIFAR10(torchvision.datasets.CIFAR10):
    # CIFAR10 Dataset for Albumentations
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return img, target


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


def d_logistic(d_logit_real, d_logit_fake):
    return (F.softplus(-d_logit_real) + F.softplus(d_logit_fake)).mean()


def g_logistic(d_logit_fake):
    return F.softplus(-d_logit_fake).mean()


def train(
    data_root='./data',
    logs_root='./logs',
    save_period=1,
    dim_x=128,
    dim_z=100,
    learning_rate=1e-3,
    beta1=0.5,
    beta2=0.999,
    epochs=50,
    batch_size=64,
    num_workers=1,
    pin_memory=True,
):
    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(
        os.path.join(logs_root, 'cifar10_gan.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    T_train = []
    T_train.append(albumentations.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    T_train.append(albumentations.pytorch.ToTensorV2())
    T_train = albumentations.Compose(T_train)

    train_dataset = CIFAR10(
        root=data_root, train=True, transform=T_train, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    gen = Generator(dim_x, dim_z).to(device)
    dis = Discriminator(dim_x).to(device)

    optimizer_g = torch.optim.Adam(
        gen.parameters(), lr=learning_rate, betas=(beta1, beta2))
    optimizer_d = torch.optim.Adam(
        dis.parameters(), lr=learning_rate, betas=(beta1, beta2))

    logger.info('%12s %12s %12s %12s %12s' %
                ('epoch', 'total_time', 'train_time', 'loss_d', 'loss_g'))

    time_total = 0
    for epoch in range(epochs):
        t0 = time.time()

        losses_d = 0
        losses_g = 0

        for i, (x, y) in enumerate(train_loader):
            real_img = x.to(device, non_blocking=True)
            dis.zero_grad(set_to_none=True)
            real_out = dis(real_img).view(-1)
            latent_z = torch.randn(x.size(0), dim_z, 1, 1)
            latent_z = latent_z.to(device, non_blocking=True)
            fake_img = gen(latent_z)
            fake_out = dis(fake_img.detach()).view(-1)

            loss_d = d_logistic(real_out, fake_out)
            loss_d.backward()
            optimizer_d.step()

            gen.zero_grad(set_to_none=True)
            fake_out = dis(fake_img).view(-1)

            loss_g = g_logistic(fake_out)
            loss_g.backward()
            optimizer_g.step()

            losses_d += loss_d.detach()
            losses_g += loss_g.detach()

        losses_d /= len(train_loader)
        losses_g /= len(train_loader)

        t1 = time.time()
        time_train = t1 - t0

        time_total += time_train

        log = '%12d %12.4f %12.4f %12.4f %12.4f' % (
            epoch + 1, time_total, time_train, losses_d, losses_g)

        logger.info(log)

        if (epoch + 1) % save_period == 0:
            gen_path = os.path.join(model_path, 'G_%03d.pth' % (epoch+1))
            logger.info(gen_path)
            gen.state_dict()
            torch.save(gen.state_dict(), gen_path)

    log = '| %12.4f | %12.4f | %12.4f |' % (time_total, losses_d, losses_g)
    logger.info('')
    logger.info(log)


def test(
    logs_root='./logs',
    dim_x=128,
    dim_z=100,
):
    image_path = os.path.join(logs_root, 'images')
    os.makedirs(image_path, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    names = sorted(os.listdir(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = Generator(dim_x, dim_z).to(device)
    gen = gen.eval()

    w, h = 20, 5
    size = w * h
    latent_z = torch.randn(size, dim_z, 1, 1, device=device)

    pack = []
    for name in names:
        gen_path = os.path.join(model_path, name)
        gen.load_state_dict(torch.load(gen_path))
        fake_img = gen(latent_z)
        fake_img = fake_img.detach().cpu().numpy()
        fake_img = np.transpose(fake_img, (0, 2, 3, 1))
        fake_img = fake_img * 0.5 + 0.5
        fake_img = fake_img * 255.0
        fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
        images = np.reshape(fake_img, (w, h*32, 32, 3))
        images = np.concatenate(images, axis=1)
        image_file = os.path.join(image_path, name + '.png')
        print(image_file)
        imageio.imwrite(image_file, images)
        pack.append(images)

    gif_file = os.path.join(logs_root, 'images.gif')
    imageio.mimsave(gif_file, pack)


if __name__ == '__main__':
    train()
    test()
