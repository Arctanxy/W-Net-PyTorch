#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 预加载数据到内存中，提高训练速度
from torchvision import transforms
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm
import os.path as osp
import pandas as pd
from PIL import Image
import random

from src.config import conf
from src.data import generate_img


class PreLoadData(Dataset):
    def __init__(self, font_size=60, transform=None, subset="train"):
        if subset == "train":
            font_folder = osp.join(conf.folder, "fonts", "train_fonts")
        elif subset == "val":
            font_folder = osp.join(conf.folder, "fonts", "val_fonts")
        else:
            print("Subset could not find. ")

        self.fonts = (
            glob(osp.join(font_folder, "*.ttf"))
            + glob(osp.join(font_folder, "*.TTF"))
            + glob(osp.join(font_folder, "*.ttc"))
            + glob(osp.join(font_folder, "*.TTC"))
            + glob(osp.join(font_folder, "*.otf"))
        )
        self.fonts = sorted(self.fonts)  # 排序以便编号
        self.fonts = self.fonts[: conf.num_fonts]  # 调试模型的时候只用一部分
        self.font_size = font_size
        characters = pd.read_excel(osp.join(conf.folder, "res", "3500常用汉字.xls"))
        self.characters = list(characters["hz"].values)
        self.characters = self.characters[: conf.num_chars]  # 调试模型的时候只用一部分

        self.protype_font = osp.join(conf.folder, "fonts", "MSYHBD.TTF")  # 基准字体
        self.protype_paths = [
            osp.join(conf.folder, "data", "{}_MSYHBD.jpg".format(item))
            for item in self.characters
        ]
        self.protype_imgs = [None for i in range(len(self.characters))]
        for i in range(len(self.characters)):
            if osp.exists(self.protype_paths[i]):
                img = Image.open(self.protype_paths[i])
            else:
                img = generate_img(self.characters[i], self.protype_font, 60)
            self.protype_imgs[i] = img

        self.font_names = [
            osp.basename(font).split(".")[0] for font in self.fonts
        ]
        self.img_path_dict = {
            j: [
                osp.join(
                    conf.folder,
                    "data",
                    "{}_{}.jpg".format(self.characters[i], self.fonts[j]),
                )
                for i in range(conf.num_chars)
            ]
            for j in range(len(self.fonts))
        }
        self.img_dict = {
            j: [None for i in range(conf.num_chars)]
            for j in range(len(self.fonts))
        }
        print("loading data ...")
        for k, v in tqdm(
            self.img_path_dict.items(), total=len(self.img_path_dict)
        ):
            for i in range(len(v)):
                if osp.exists(v[i]):
                    img = Image.open(v[i])
                else:
                    img = generate_img(self.characters[i], self.fonts[k], 60)
                self.img_dict[k][i] = img

        if transform is None:
            self.transform = transforms.Compose(
                [transforms.Resize((64, 64)), transforms.ToTensor()]
            )
        else:
            self.transform = transform

    def __getitem__(self, index):
        font_index = index // conf.num_chars
        char_index = index % conf.num_chars
        protype_img = self.protype_imgs[char_index]
        real_img = self.img_dict[font_index][char_index]

        style_char_index = random.randint(0, len(self.characters) - 1)
        style_img = self.img_dict[font_index][style_char_index]

        protype_img = self.transform(protype_img)
        real_img = self.transform(real_img)
        style_img = self.transform(style_img)

        return (
            protype_img,
            char_index,
            style_img,
            font_index,
            style_char_index,
            real_img,
        )

    def __len__(self):
        return len(self.fonts) * conf.num_chars - 1


if __name__ == "__main__":
    d = PreLoadData()

