from torch import nn
import torch
from torch import autograd
import torch.nn.functional as F
from src.config import conf


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, x_fake, x_real):
        N = x_real.size(0)
        smooth = 1

        input_flat = x_fake.view(N, -1)
        target_flat = x_real.view(N, -1)

        intersection = input_flat * target_flat
        loss = (
            2
            * (intersection.sum(1) + smooth)
            / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        )
        loss = 1 - loss.sum() / N
        return loss


# refer to https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py 

def calc_gradient_penalty(D, x_real, x_fake, x1, x2):
    #print real_data.size()

    x_real.requires_grad = True 
    x_fake.requires_grad = True 
    x1.requires_grad = True 
    x2.requires_grad = True 
    alpha = torch.rand(conf.batch_size, 1)
    alpha = alpha.expand(x_real.size())
    alpha = alpha.to(conf.device)

    interpolates = alpha * x_real + ((1 - alpha) * x_fake)

    interpolates = interpolates.to(conf.device)

    disc_interpolates = D(interpolates, x1, x2)[:3] # 最后一个是中间层feature，不需要
    
    # print(x_real.requires_grad, x_fake.requires_grad, x1.requires_grad, x2.requires_grad)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=[interpolates,x1,x2],
                              grad_outputs=[torch.ones(disc_interpolates[0].size()).to(conf.device),
                                            torch.ones(disc_interpolates[1].size()).to(conf.device),
                                            torch.ones(disc_interpolates[2].size()).to(conf.device)
                                            ],
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class GenerationLoss(nn.Module):
    def __init__(self):
        super(GenerationLoss, self).__init__()
        if conf.label_smoothing:
            self.cls_criteron = LabelSmoothing()
        else:
            self.cls_criteron = nn.CrossEntropyLoss()

    def forward(
        self,
        out,
        out_real,
        real_label,
        real_style_label,
        char_label,
        x_fake,
        x_real,
        encoder_out_real_left,
        encoder_out_fake_left,
        encoder_out_real_right,
        encoder_out_fake_right,
        cls_enc_p=None,
        cls_enc_s=None,
    ):
        self.real_fake_loss = conf.alpha * nn.BCELoss()(
            out[0], real_label.float()
        )
        self.style_category_loss = conf.beta_d * self.cls_criteron(
            out[1], real_style_label
        )
        self.char_category_loss = conf.beta_d * self.cls_criteron(
            out[2], char_label
        )

        if conf.reconstruction_loss_type == "dice":
            self.reconstruction_loss = conf.lambda_l1 * DiceLoss()(
                x_fake, x_real
            )
        elif conf.reconstruction_loss_type == "l1":
            self.reconstruction_loss = conf.lambda_l1 * nn.L1Loss()(
                x_fake, x_real
            )

        # 原论文里面使用训练好的vgg字符分类网络的中间特征来做
        # 这里为了省事，直接用的Discriminator的中间层特征
        self.reconstruction_loss2 = conf.lambda_phi * (
            nn.MSELoss()(out[3][0], out_real[3][0])
            + nn.MSELoss()(out[3][1], out_real[3][1])
            + nn.MSELoss()(out[3][2], out_real[3][2])
            + nn.MSELoss()(out[3][3], out_real[3][3])
        )

        self.left_constant_loss = conf.phi_p * nn.MSELoss()(
            encoder_out_real_left, encoder_out_fake_left
        )
        self.right_constant_loss = conf.phi_r * nn.MSELoss()(
            encoder_out_real_right, encoder_out_fake_right
        )
        self.content_category_loss = conf.beta_p * self.cls_criteron(
            cls_enc_p, char_label
        )  # category loss for content prototype encoder
        self.style_category_loss = conf.beta_r * self.cls_criteron(
            cls_enc_s, real_style_label
        )
        return (
            self.real_fake_loss
            + self.style_category_loss
            + self.char_category_loss
            + self.reconstruction_loss
            + self.reconstruction_loss2
            + self.left_constant_loss
            + self.right_constant_loss
            + self.content_category_loss
            + self.style_category_loss
        )


class DiscriminationLoss(nn.Module):
    def __init__(self):
        super(DiscriminationLoss, self).__init__()
        if conf.label_smoothing:
            self.cls_criteron = LabelSmoothing()
        else:
            self.cls_criteron = nn.CrossEntropyLoss()


    def forward(
        self,
        out_real,
        out_fake,
        real_label,
        fake_label,
        real_style_label,
        fake_style_label,
        char_label,
        fake_char_label,
        cls_enc_p=None,
        cls_enc_s=None,
        D = None,x_real=None,x_fake=None, x1 = None, x2 = None
    ):
        self.real_loss = conf.alpha * nn.BCELoss()(
            out_real[0], real_label.float()
        )  # fake or real loss
        self.fake_loss = conf.alpha * nn.BCELoss()(
            out_fake[0], fake_label.float()
        )  # fake or real loss
        self.real_style_loss = conf.beta_d * self.cls_criteron(
            out_real[1], real_style_label
        )  # style category loss
        self.fake_style_loss = conf.beta_d * self.cls_criteron(
            out_fake[1], fake_style_label
        )  # style category loss
        self.real_char_category_loss = conf.beta_d * self.cls_criteron(
            out_real[2], char_label
        )  # char category loss
        self.fake_char_category_loss = conf.beta_d * self.cls_criteron(
            out_fake[2], fake_char_label
        )  # char category loss
        self.content_category_loss = conf.beta_p * self.cls_criteron(
            cls_enc_p, char_label
        )
        self.style_category_loss = conf.beta_r * self.cls_criteron(
            cls_enc_s, real_style_label
        )
        if D:
            self.gradient_penalty = conf.alpha_GP * calc_gradient_penalty(D,x_real, x_fake, x1, x2)
        else:
            self.gradient_penalty = 0.0

        return 0.5 * (
            self.real_loss
            + self.fake_loss
            + self.real_style_loss
            + self.fake_style_loss
            + self.real_char_category_loss
            + self.fake_char_category_loss
            + self.content_category_loss
            + self.style_category_loss 
            + self.gradient_penalty 
        )
