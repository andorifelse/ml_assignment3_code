This is my Machine Learning coures assignment3.

Project is based on [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).

Special thanks to the original author for making the source code publicly available.

训练使用本地 `/home/wzc/nas/home/Dataset/CIFAR10/CIFAR10/{train,test}`，不会下载数据。

从头训练 DPN92：

```bash
python main.py --arch DPN92 --run-name dpn92_aug_v1
```

默认使用 CIFAR-10 AutoAugment、Mixup（alpha=0.2）、5 轮学习率预热及余弦衰减，共训练 300 轮。增强只用于训练集；Mixup 开启后的 `Mixup Acc` 是针对两个来源标签加权的准确率，不能直接与原来的训练准确率比较。新配置是否提高测试准确率需要通过训练验证。

每个实验的检查点独立保存在 `checkpoint/<网络>/<实验名>/`：`ckpt.pth` 保存测试准确率最高的模型，`last.pth` 保存最新一轮及优化器、学习率调度器状态。实验名已存在时，新训练会拒绝覆盖。

中断后恢复最新一轮（训练参数沿用该实验保存的配置，`--epochs` 表示总轮数）：

```bash
python main.py --arch DPN92 --run-name dpn92_aug_v1 --resume
```

旧版检查点只有模型权重，使用 `--init-from` 在新实验中微调；将下方的 `OLD_RUN` 替换为旧实验目录名。这会新建优化器和学习率计划：

```bash
python main.py --arch DPN92 --run-name dpn92_finetune_v1 \
  --init-from checkpoint/DPN92/OLD_RUN/ckpt.pth \
  --lr 0.01 --epochs 100
```

复现原来的增强和训练计划以作比较：

```bash
python main.py --arch DPN92 --run-name dpn92_baseline \
  --auto-augment none --mixup-alpha 0 --warmup-epochs 0 --epochs 200
```
