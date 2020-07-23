# 基于Wnet的字体生成

原论文：[W-Net: One-Shot Arbitrary-Style Chinese Character Generation with Deep Neural Networks](https://www.researchgate.net/publication/329007858_W-Net_One-Shot_Arbitrary-Style_Chinese_Character_Generation_with_Deep_Neural_Networks_25th_International_Conference_ICONIP_2018_Siem_Reap_Cambodia_December_13-16_2018_Proceedings_Part_V)


与原论文的差别
---

1. Discriminator中没有使用LayerNorm，而是用了BatchNorm
2. loss中没有加梯度惩罚项
3. 没有额外再训练一个VGG分类模型，而是用Discriminator替代

TODO:
---

- [x] label smoothing
- [ ] 在G的训练和测试阶段都添加dropout
- [ ] 使用LeaklyReLU替代ReLU
- [ ] 在Discriminator中使用LayerNorm
- [ ] 每个batch中使用同一种字体（据说可以使训练变得更简单）
- [ ] 监控训练中的梯度变化
- [ ] 添加梯度惩罚
- [ ] 历史均值
