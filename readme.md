# 基于Wnet的字体生成

原论文：[W-Net: One-Shot Arbitrary-Style Chinese Character Generation with Deep Neural Networks](https://www.researchgate.net/publication/329007858_W-Net_One-Shot_Arbitrary-Style_Chinese_Character_Generation_with_Deep_Neural_Networks_25th_International_Conference_ICONIP_2018_Siem_Reap_Cambodia_December_13-16_2018_Proceedings_Part_V)

算法思路简介：
1. 图中左侧的分支用于提取汉字结构信息
2. 图中右侧的分支用于提取字体风格信息 
3. 字体风格特征只选用了较深层网络得到的特征图
4. 采用对抗训练的方式得到与真实汉字相近的图片 


现阶段的训练结果：

原始字体（黑体加粗）protype
---
![](./img/src.png)

目标字体（一个batch里面混合了多种字体）real
---
![](./img/target.png)

生成字体 fake
---
![](./img/out.png)


与原论文的差别
---

1. Discriminator中没有使用LayerNorm，而是用了BatchNorm
2. loss中没有加梯度惩罚项
3. 没有额外再训练一个VGG分类模型，而是用Discriminator替代

TODO:
---

tricks:

- [x] label smoothing
- [ ] 在G的训练和测试阶段都添加dropout
- [ ] 使用LeaklyReLU替代ReLU
- [ ] 在Discriminator中使用LayerNorm
- [ ] 每个batch中使用同一种字体（据说可以使训练变得更简单）
- [ ] 监控训练中的梯度变化
- [x] 添加梯度惩罚
- [ ] 历史均值
- [ ] 模型推理代码
