import os
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_image_files(root):
    paths = []
    for (root, dirs, files) in os.walk(root):
        for f in files:
            if f[-3:] in ['jpg', 'png']:
                path = os.path.join(root, f)
                paths.append(path)
    return paths


def prepare_data(root, save_path='./data/celeba_hq_32', size=32):
    paths = get_image_files(root)
    os.makedirs(save_path, exist_ok=True)
    for path in tqdm(paths):
        img = Image.open(path)
        img = img.resize((size, size), Image.LANCZOS)
        name = os.path.basename(path)
        save_file = os.path.join(save_path, name + '.png')
        img.save(save_file)


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        paths = get_image_files(root)
        data = []
        for path in tqdm(paths):
            img = Image.open(path)
            arr = np.asarray(img)
            data.append(arr)
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return img
