import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np
from utils.image_utils import load_img


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, mask_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        tar_files = sorted(os.listdir(rgb_dir))
        mask_files = sorted(os.listdir(mask_dir))

        self.tar_filenames = [os.path.join(rgb_dir, x) for x in tar_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(mask_dir, x) for x in mask_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)
        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        tar_path = self.tar_filenames[index_]
        mask_path = self.mask_filenames[index_]

        tar_img = Image.open(tar_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        if padw != 0 or padh != 0:
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')
            mask = TF.pad(mask, (0, 0, padw, padh), padding_mode='reflect')

        tar_img = TF.to_tensor(tar_img)
        mask = TF.to_tensor(mask)

        hh, ww = tar_img.shape[1], tar_img.shape[2]
        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]
        mask = mask[:, rr:rr + ps, cc:cc + ps]

        if aug == 1:
            tar_img = tar_img.flip(1)
            mask = mask.flip(1)
        elif aug == 2:
            tar_img = tar_img.flip(2)
            mask = mask.flip(2)
        elif aug == 3:
            tar_img = torch.rot90(tar_img, dims=(1, 2))
            mask = torch.rot90(mask, dims=(1, 2))
        elif aug == 4:
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
            mask = torch.rot90(mask, dims=(1, 2), k=2)
        elif aug == 5:
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
            mask = torch.rot90(mask, dims=(1, 2), k=3)

        inp_img = tar_img * (1. - mask)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, mask, filename


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, mask_dir, img_options=None):
        super(DataLoaderVal, self).__init__()

        tar_files = sorted(os.listdir(rgb_dir))
        mask_files = sorted(os.listdir(mask_dir))

        self.tar_filenames = [os.path.join(rgb_dir, x) for x in tar_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(mask_dir, x) for x in mask_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        tar_path = self.tar_filenames[index_]
        mask_path = self.mask_filenames[index_]

        tar_img = Image.open(tar_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        mul = 16
        w, h = tar_img.size
        H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
        padh = H - h if h % mul != 0 else 0
        padw = W - w if w % mul != 0 else 0
        tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')
        mask = TF.pad(mask, (0, 0, padw, padh), padding_mode='reflect')

        tar_img = TF.to_tensor(tar_img)
        mask = TF.to_tensor(mask)

        inp_img = tar_img * (1. - mask)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, mask, filename


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, mask_dir):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        mask_files = sorted(os.listdir(mask_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(mask_dir, x) for x in mask_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index]
        path_mask = self.mask_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]

        inp = Image.open(path_inp).convert('RGB')
        mask = Image.open(path_mask).convert('L')

        inp = TF.to_tensor(inp)
        mask = TF.to_tensor(mask)

        return inp, mask, filename
