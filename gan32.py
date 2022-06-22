import torch.nn as nn
from utils import *
from pytorch_fid.fid_score import calculate_fid_given_paths


def Generator(dim_x=128, dim_z=100):
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


def Discriminator(dim_x=128):
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


def main():
    dim_x = 128
    dim_z = 100

    gen = Generator(dim_x, dim_z)
    dis = Discriminator(dim_x)

    cifar10 = False
    if cifar10:
        logs_root = './logs/cifar10'
        data_root = './data/cifar10'
        dataset = get_cifar10_dataset(data_root)
    else:
        logs_root = './logs/celeba_hq_32'
        data_root = './data/celeba_hq_32'
        #prepare_data('../data/celeba_hq', data_root, 32)
        dataset = get_custum_dataset(data_root)

    train(logs_root, dataset, gen, dis, dim_z, epochs=200)
    test(logs_root, gen, dim_z)


def evaluate():
    dim_x = 128
    dim_z = 100

    logs_root = './logs/celeba_hq_32'
    data_root = './data/celeba_hq_32'
    image_path = os.path.join(logs_root, 'generated')
    os.makedirs(image_path, exist_ok=True)
    model_path = './logs/celeba_hq_32/models/G_200.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = Generator(dim_x, dim_z)
    gen.load_state_dict(torch.load(model_path))
    gen = gen.eval()
    gen = gen.to(device)

    size = 30000
    batch_size = 100
    steps = size // batch_size

    with torch.inference_mode():
        for step in range(steps):
            latent_z = torch.randn(batch_size, dim_z, 1, 1, device=device)
            fake_img = gen(latent_z)
            fake_img = fake_img.detach().cpu().numpy()
            fake_img = np.transpose(fake_img, (0, 2, 3, 1))
            fake_img = fake_img * 0.5 + 0.5
            fake_img = fake_img * 255.0
            fake_img = np.clip(fake_img, 0, 255).astype(np.uint8)
            for i in range(batch_size):
                image_file = os.path.join(
                    image_path, '%06d.png' % (step * batch_size + i))
                print(image_file)
                imageio.imwrite(image_file, fake_img[i])

    batch_size = 50
    num_workers = 4
    dims = 2048
    paths = [data_root, image_path]
    fid_value = calculate_fid_given_paths(
        paths=paths,
        batch_size=batch_size,
        device=device,
        dims=dims,
        num_workers=num_workers,
    )

    print('FID: ', fid_value)


if __name__ == '__main__':
    main()
    evaluate()
