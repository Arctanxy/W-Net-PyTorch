# 基于Wnet的字体生成




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
2. 没有额外再训练一个VGG分类模型，而是用Discriminator替代

TODO:
---

待添加的一些tricks:

- [x] label smoothing
- [ ] 在G的训练和测试阶段都添加dropout
- [ ] 使用LeaklyReLU替代ReLU
- [ ] 在Discriminator中使用LayerNorm
- [ ] 每个batch中使用同一种字体（据说可以使训练变得更简单）
- [ ] 监控训练中的梯度变化
- [x] 添加梯度惩罚
- [ ] 历史均值
- [ ] 模型推理代码
- [ ] 常数loss与重建loss2的最后一个特征重复，需要去掉
