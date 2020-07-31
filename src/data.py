# 使用字体生成图片

from PIL import ImageFont, ImageDraw, Image
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os.path as osp
import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from torchvision import transforms

try:
    from src.config import conf
except:
    from config import conf


def generate_img(word, font_file, font_size, w=80):
    file_name = osp.join(
        conf.folder,
        conf.img_folder,
        "{}_{}.jpg".format(word, osp.basename(font_file).split(".")[0]),
    )
    if not osp.exists(osp.join(conf.folder, conf.img_folder)):
        os.mkdir(osp.join(conf.folder, conf.img_folder))
    # print(file_name)
    # print(word, font_file)
    if osp.exists(file_name):
        if conf.reconstruction_loss_type == "dice":
            return Image.open(file_name).convert("1")
        else:
            # img = Image.open(file_name)
            # img.load()
            img_ = Image.open(file_name)
            img = img_.copy()
            img_.close()
            return img
    # print("font file\t", font_file)
    font = ImageFont.truetype(font_file, font_size)
    img = Image.new(mode="L", size=(w, w), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), word, fill=0, font=font)
    # img.save("wo_{}.jpg".format(font_file.split("/")[2]))
    # 裁剪
    img_array = np.array(img)
    coordinates = np.where(img_array == 0)
    # todo: 有些字体连常用字都不支持，需要跳过或者预筛选
    if len(coordinates[0]) == 0:
        print("{}字体不支持{}".format(font_file, word))
        return img
    min_x, max_x = np.min(coordinates[0]), np.max(coordinates[0])
    min_y, max_y = np.min(coordinates[1]), np.max(coordinates[1])
    width = max_y - min_y
    height = max_x - min_x
    side = max(width, height)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    min_x = max(int(center_x - side / 2), 0)
    min_y = max(int(center_y - side / 2), 0)
    max_x = min(int(center_x + side / 2), w)
    max_y = min(int(center_y + side / 2), w)
    img_array = img_array[min_x:max_x, min_y:max_y]
    img = Image.fromarray(img_array).resize((64, 64))
    # img.save("wo_{}.jpg".format(font_file.split("/")[2]))
    # 使用diceloss需要把图片搞成二值图片，按分割图的思路来做
    if conf.reconstruction_loss_type == "dice":
        img = img.convert("1")
    img.save(file_name)
    return img


transform = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor()]
)


class CHNData(Dataset):
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
        self.font_size = font_size
        characters = pd.read_excel("3500常用汉字.xls")
        self.characters = list(characters["hz"].values)
        self.characters = self.characters[: conf.num_chars]  # 调试模型的时候只用一部分
        if conf.num_chars < 3500:
            print("使用文字:{}".format(self.characters))
        self.protype_font = osp.join(conf.folder, "fonts", "MSYHBD.TTF")  # 基准字体
        self.fonts = [f for f in self.fonts if "MSYHBD" not in f]  # 风格字体
        self.fonts = self.fonts[: conf.num_fonts]  # 调试模型的时候只用一部分

        if conf.num_fonts < 28:
            print("使用字体：{},共计{}个".format(self.fonts, len(self.fonts)))
        # print("fonts", len(self.fonts), self.fonts)
        if transform is None:
            self.transform = transforms.Compose(
                [transforms.Resize((64, 64)), transforms.ToTensor()]
            )
        else:
            self.transform = transform

    def __getitem__(self, index):
        protype_img = generate_img(
            self.characters[index % conf.num_chars],
            self.protype_font,
            self.font_size,
        )
        style_indices = random.randint(0, len(self.fonts) - 1)
        font = self.fonts[style_indices]
        style_character_index = random.randint(0, len(self.characters) - 1)
        style_character = self.characters[style_character_index]
        style_img = generate_img(style_character, font, self.font_size)
        real_img = generate_img(
            self.characters[index % conf.num_chars], font, self.font_size
        )
        protype_img = self.transform(protype_img)
        style_img = self.transform(style_img)
        real_img = self.transform(real_img)
        # 基准图片， 基准图片的字符标签， 风格图片， 风格图片的风格标签， 风格图片的字符标签
        return (
            protype_img,
            index % conf.num_chars,
            style_img,
            style_indices,
            style_character_index,
            real_img,
        )

    def __len__(self):
        # return len(self.characters)
        # return conf.num_samples
        # return min(conf.num_samples, conf.num_chars)
        return min(conf.num_samples, len(self.fonts) * conf.num_chars - 1)


if __name__ == "__main__":
    # fonts = ["MSYH.TTF", "SIMHEI.TTF", "SIMFANG.TTF", "MSYHBD.TTF", "SIMSUN.TTC"]
    # fonts = ["./fonts/" + f for f in fonts]
    # for f in fonts:
    #     generate_img(f, 60)
    d = CHNData()
    dl = DataLoader(d, batch_size=16, shuffle=True)
    for (
        protype_img,
        index,
        style_img,
        style_indices,
        style_character_index,
        real_img,
    ) in d:
        print(
            type(protype_img),
            index,
            type(style_img),
            style_indices,
            style_character_index,
            type(real_img),
        )
        break
