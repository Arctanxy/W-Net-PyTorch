# 训练GAN的常用技巧

## Batch normalization

在绝大部分的深度学习任务中，Batch normalization都有比较好的效果

Batch normalization对Generator的作用尚有争议，有研究认为Batch normalization在Generator中有负面作用（Gulrajani et al., 2017, http://arxiv.org/abs/1704.00028.）

不过一般都认为Batch normalization对Discriminator有积极作用（“Tutorial on Generative Adversarial Networks—GANs in the Wild,” by Soumith Chintala, 2017, https://www.youtube.com/watch?v=Qc1F3-Rblbw.
）

> 在Generator的loss中使用了梯度惩罚的情况下，Discriminator尽量避免使用Batch normalization，可以考虑使用Layer normalization、Weight Normalization或者Instance Normalization等。

## 梯度惩罚

GAN的对抗训练机制让Generator和Discriminator的梯度极不稳定，很容易出现训练发散的情况。

因此需要对梯度进行限制，早期研究中常常会使用梯度剪裁来限制梯度变化，但是简单的剪裁可能会带来梯度消失或者爆炸的情况出现。

近些年来很多关于GAN的论文都使用到了名为梯度惩罚的技术，即将模型对于输入的梯度作为loss中的惩罚项，

使得模型输入有微小变化的时候，网络权重不会产生太大的变化。


## 优先训练Discriminator

这个策略下大致有如下三种不同的实现方式：

1. 在Generator开始训练之前，先训练一个能判别真假的Discriminator；
2. 每训练n（n>=1）次Discriminator，训练一次Generator；
3. 在Discriminator中使用更大的学习率（Heusel, Martin et al. “GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.” NIPS (2017)）


## 避免梯度稀疏以及信息丢失

ReLU或者MaxPool产生的稀疏梯度会造成训练困难，生成任务与一般的分类回归任务不同的是，生成任务需要非常完整的细节信息，因此，这些操作中产生的信息丢失会影响到Generator的训练。

因此，在GAN中，因尽量避免使用池化层（MaxPool、AvgPool等），可以使用Leaky-ReLU替代ReLU。


## 标签平滑或者添加噪声

在Discriminator和Generator的loss中都有不少的分类loss，使用标签平滑或者合理地对标签添加噪声都可以降低训练难度。

## 使用更多的标签信息

在训练过程中，除了图片的真假信息外，如果数据集中有其他信息，尽量利用起来，能够提升模型训练效果。

## 指数平均参数

通过对不同epoch下的参数求指数平均可以让训练过程变得更加平稳（Yazici, Yasin et al. “The Unusual Effectiveness of Averaging in GAN Training.” CoRRabs/1806.04498 (2018): n. pag.）
不过指数平均中有一个超参需要设置，不想调这个超参的话，直接只用滑动平均参数也可以获得不错的效果。


## 利用分类网络建立图片的重建loss 

在Generator的损失函数中，通常会加入一个重建损失，用于评估生成图片和真实图片之间的差距。

在一些对生成图片的细节要求不高的任务中，可以直接使用L1Loss作为重建损失,

为了得到更细致的生成结果，可以i利用分类的特征提取能力，将生成图片和真实图片在分类网络中得到的特征图之间的差距加入到重建损失中。


末尾贴一个我自己复现的名为W-Net的项目

https://github.com/Arctanxy/W-Net-PyTorch 

目前使用过，且有效的训练技巧有：梯度惩罚、标签平滑、在Discriminator中使用更大的学习率、利用分类网络建立重建loss。

后面会陆续补充其他训练技巧及相应的代码实现。
