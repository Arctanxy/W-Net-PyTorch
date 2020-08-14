import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
import torchvision

from src.config import conf
from src.model import WNet, Discriminator, CLSEncoderS, ClSEncoderP
from src.loss_func import GenerationLoss, DiscriminationLoss


class Trainer(object):
    def __init__(self, train_loader, valid_loader):
        self.init_model()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = SummaryWriter("./log/" + conf.title + conf.time_stamp)

    def train_one_epoch(self, epoch):
        epoch_reconstruction_loss = 0.0
        epoch_lgs = 0.0
        epoch_lds = 0.0
        cnt = 0
        for (
            protype_img,
            index,
            style_img,
            style_indices,
            style_character_index,
            real_img,
        ) in tqdm(self.train_loader, total=len(self.train_loader)):
            if cnt == len(self.train_loader):
                break
            cnt += 1
            x1 = protype_img.to(conf.device)
            x2 = style_img.to(conf.device)
            x_real = real_img.to(conf.device)
            real_style_label = style_indices.to(conf.device)  # 真实的风格标签
            fake_style_label = torch.tensor(
                [
                    conf.num_fonts for i in range(x1.shape[0])
                ]  # drop_last = True的时候可以把用range(conf.batch_size)
            ).to(
                conf.device
            )  # 假的风格标签
            char_label = index.to(conf.device)  # 真实的字形标签
            fake_char_label = torch.tensor(
                [conf.num_chars for i in range(x1.shape[0])]
            ).to(
                conf.device
            )  # 假的字形标签
            real_label = torch.tensor([1 for i in range(x1.shape[0])]).to(
                conf.device
            )  # 真样本标签
            fake_label = torch.tensor([0 for i in range(x1.shape[0])]).to(
                conf.device
            )  # 假样本标签

            self.optimizer_G.zero_grad()
            x_fake, lout, rout = self.G(x1, x2)
            out = self.D(x_fake, x1, x2)
            out_real_ = self.D(x_real, x1, x2)  # 加个下划线，避免跟后面重名

            # 两边encoder之后接一个分类器
            cls_enc_p = self.CLSP(lout.view(-1, 512))
            cls_enc_s = self.CLSS(rout.view(-1, 512))

            # fake和real通过两个encoder得到的向量应该相同
            if conf.multi_gpus:
                encoder_out_real_left = self.G.module.left(x_real)[5]
                encoder_out_real_right = self.G.module.right(x_real)[5]
                encoder_out_fake_left = self.G.module.left(x_fake)[5]
                encoder_out_fake_right = self.G.module.right(x_fake)[5]
            else:
                encoder_out_real_left = self.G.left(x_real)[5]
                encoder_out_real_right = self.G.right(x_real)[5]
                encoder_out_fake_left = self.G.left(x_fake)[5]
                encoder_out_fake_right = self.G.right(x_fake)[5]
            criterion_G = GenerationLoss()
            L_G = criterion_G(
                out,
                out_real_,
                real_label,
                real_style_label,
                char_label,
                x_fake,
                x_real,
                encoder_out_real_left,
                encoder_out_fake_left,
                encoder_out_real_right,
                encoder_out_fake_right,
                cls_enc_p,
                cls_enc_s,
            )
            epoch_reconstruction_loss += (
                criterion_G.reconstruction_loss.item() / conf.lambda_l1
            )
            epoch_lgs += L_G.item()

            L_G.backward(retain_graph=True)  # 为何为提示重复backward
            self.optimizer_G.step()

            self.optimizer_D.zero_grad()
            out_real = self.D(x_real, x1, x2)
            out_fake = self.D(x_fake.detach(), x1, x2)
            cls_enc_p = self.CLSP(lout.view(-1, 512).detach())
            cls_enc_s = self.CLSS(rout.view(-1, 512).detach())

            # 真假分类损失，风格分类损失，两个encoder提取到的特征质量损失
            if conf.label_smoothing:
                L_D = DiscriminationLoss()(
                    out_real,
                    out_fake,
                    real_label,
                    fake_label,
                    real_style_label,
                    fake_style_label,
                    char_label,
                    fake_char_label,
                    cls_enc_p,
                    cls_enc_s,
                    self.D,
                    x_real,
                    x_fake.detach(),
                    x1,
                    x2,
                )
            else:
                L_D = DiscriminationLoss()(
                    out_real,
                    out_fake,
                    real_label,
                    fake_label,
                    real_style_label,
                    fake_style_label,
                    char_label,
                    fake_char_label,
                    cls_enc_p,
                    cls_enc_s,
                )
            epoch_lds += L_D.item()
            L_D.backward()
            self.optimizer_D.step()
        epoch_reconstruction_loss /= len(self.train_loader)
        epoch_lgs /= len(self.train_loader)
        epoch_lds /= len(self.train_loader)
        fake_image = torchvision.utils.make_grid(x_fake)
        real_image = torchvision.utils.make_grid(x_real)
        src_image = torchvision.utils.make_grid(x1)
        self.writer.add_image("fake", fake_image, epoch)
        self.writer.add_image("real", real_image, epoch)
        self.writer.add_image("src", src_image, epoch)
        self.writer.add_scalars(
            "losses",
            {
                "G_LOSS": epoch_lgs,
                "D_LOSS": epoch_lds,
                "train_reconstruction": epoch_reconstruction_loss,
            },
            epoch,
        )
        if epoch % conf.save_epoch == 0:
            eval_loss = self.eval_model()
            self.writer.add_scalars(
                "losses", {"eval_reconstruction": eval_loss}, epoch
            )
            print("Eval Loss: {}".format(eval_loss))
            self.save_model(
                "model_epoch_{}_loss_{}.pth".format(epoch, eval_loss)
            )
        # scheduler_G.step()
        # scheduler_D.step()

    def init_model(self):
        self.G = WNet().to(conf.device)
        self.D = Discriminator(conf.num_fonts + 1, conf.num_chars + 1).to(
            conf.device
        )
        # 两个encoder后面的外接分类器
        self.CLSP = ClSEncoderP(conf.num_chars + 1).to(conf.device)
        self.CLSS = CLSEncoderS(conf.num_fonts + 1).to(conf.device)

        self.optimizer_G = optim.Adam(
            self.G.parameters(),
            lr=conf.init_lr,
            betas=(conf.beta_1, conf.beta_2),
            weight_decay=conf.weight_decay,
        )
        # scheduler_G = ExponentialLR(optimizer_G, 0.99)

        self.optimizer_D = optim.Adam(
            chain(
                self.D.parameters(),
                self.CLSP.parameters(),
                self.CLSS.parameters(),
            ),
            lr=conf.init_lr,
            betas=(conf.beta_1, conf.beta_2),
            weight_decay=conf.weight_decay,
        )
        # scheduler_D = ExponentialLR(optimizer_D, 0.99)
        if conf.ckpt is not None:
            print("Loading model from {}".format(conf.ckpt))
            # 增加font和char的数量不影响G的结构，所以G可以在不同试验中重复使用
            params = torch.load(conf.ckpt, map_location="cpu")
            try:
                self.G.load_state_dict(params["G"])
                print("G 加载成功...")
            except Exception as e:
                print("G 加载失败...")
            try:
                self.D.load_state_dict(params["D"])
                self.CLSP.load_state_dict(params["CLSP"])
                self.CLSS.load_state_dict(params["CLSS"])
                print("D,CLSP,CLSS 加载成功...")
                # 如果类别变化了，optimizer就算加载成功也会在step处报错
                self.optimizer_G.load_state_dict(params["optimizer_G"])
                self.optimizer_D.load_state_dict(params["optimizer_D"])
                print("Optimizer D, G 加载成功")

            except Exception as e:
                print("D,CLSP,CLSS 加载失败...")
                print("optimizer G 加载失败 ...")
                print("optimizer D 加载失败 ...")
        else:
            print("从头开始训练...")
        if conf.multi_gpus:
            self.G = torch.nn.DataParallel(self.G, device_ids=conf.device_ids)
            self.D = torch.nn.DataParallel(self.D, device_ids=conf.device_ids)
            self.CLSP = torch.nn.DataParallel(
                self.CLSP, device_ids=conf.device_ids
            )
            self.CLSS = torch.nn.DataParallel(
                self.CLSS, device_ids=conf.device_ids
            )

    def save_model(self, check_point):
        if conf.multi_gpus:
            torch.save(
                {
                    "G": self.G.module.state_dict(),
                    "D": self.D.module.state_dict(),
                    "CLSP": self.CLSP.module.state_dict(),
                    "CLSS": self.CLSS.module.state_dict(),
                    "optimizer_G": self.optimizer_G.state_dict(),
                    "optimizer_D": self.optimizer_D.state_dict(),
                },
                check_point,
            )
        else:
            torch.save(
                {
                    "G": self.G.state_dict(),
                    "D": self.D.state_dict(),
                    "CLSP": self.CLSP.state_dict(),
                    "CLSS": self.CLSS.state_dict(),
                    "optimizer_G": self.optimizer_G.state_dict(),
                    "optimizer_D": self.optimizer_D.state_dict(),
                },
                check_point,
            )

    def eval_model(self):
        # 暂时使用重建损失来衡量G的效果好坏
        with torch.no_grad():
            self.G.eval()
            losses = []
            cnt = 0
            for (
                protype_img,
                index,
                style_img,
                style_indices,
                style_character_index,
                real_img,
            ) in tqdm(self.valid_loader, total=len(self.valid_loader)):
                if cnt == len(self.valid_loader):
                    break
                cnt += 1
                x1 = protype_img.to(conf.device)
                x2 = style_img.to(conf.device)
                x_real = real_img.to(conf.device)
                x_fake, lout, rout = self.G(x1, x2)
                reconstruction_loss = nn.L1Loss()(x_fake, x_real)
                losses.append(reconstruction_loss.item())
            eval_loss = np.mean(losses)
        self.G.train()
        return eval_loss

