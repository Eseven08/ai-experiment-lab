# -*- coding: utf-8 -*-
"""
nn_xor.py —— 从零实现的两层神经网络（解决 XOR 异或问题）

XOR（异或）是经典的"线性不可分"问题：单层感知机永远学不会它，
但加上一个隐藏层之后，神经网络就能轻松学会。

本脚本不依赖任何第三方库（只用 Python 标准库），
手写前向传播、反向传播和梯度下降，展示神经网络的核心原理。

运行方式：  python nn_xor.py
"""

import math
import random

# ---------------------------------------------------------------------------
# 1. 激活函数
# ---------------------------------------------------------------------------
def tanh(x):
    """双曲正切激活函数，把输出压缩到 (-1, 1)，用于隐藏层。"""
    return math.tanh(x)


def tanh_derivative(y):
    """tanh 的导数：1 - y^2。y 是 tanh 的输出值。"""
    return 1.0 - y * y


def sigmoid(x):
    """Sigmoid 激活函数，把输出压缩到 (0, 1)，用于输出层做二分类。"""
    if x < 0:
        # 数值稳定性处理：x 很小时直接用 exp(x) 计算，避免溢出
        e = math.exp(x)
        return e / (1.0 + e)
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(y):
    """Sigmoid 的导数：y * (1 - y)。"""
    return y * (1.0 - y)


# ---------------------------------------------------------------------------
# 2. 两层神经网络类
# ---------------------------------------------------------------------------
class TwoLayerNet:
    """
    结构：输入层(2) -> 隐藏层(hidden_size) -> 输出层(1)

    隐藏层用 tanh，输出层用 sigmoid，输出值接近 1 表示"真"，接近 0 表示"假"。
    """

    def __init__(self, input_size, hidden_size, output_size, seed=None):
        # 用独立的 Random 实例，固定种子让每次运行结果一致，且不影响全局随机状态
        rng = random.Random(seed)

        # 权重矩阵随机初始化（小数值，避免一开始就饱和）
        # W1: 输入 -> 隐藏层，形状 (hidden_size, input_size)
        # W2: 隐藏层 -> 输出层，形状 (output_size, hidden_size)
        self.W1 = [[rng.uniform(-0.5, 0.5) for _ in range(input_size)]
                   for _ in range(hidden_size)]
        self.W2 = [[rng.uniform(-0.5, 0.5) for _ in range(hidden_size)]
                   for _ in range(output_size)]

        # 偏置（bias）
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.b2 = [0.0 for _ in range(output_size)]

    # 前向传播：输入 -> 隐藏层 -> 输出层
    def forward(self, x):
        # 隐藏层：z1 = W1 @ x + b1，再经过 tanh
        z1 = [sum(w * x[i] for i, w in enumerate(row)) + b
              for row, b in zip(self.W1, self.b1)]
        h = [tanh(z) for z in z1]  # 隐藏层输出

        # 输出层：z2 = W2 @ h + b2，再经过 sigmoid
        z2 = [sum(w * h[j] for j, w in enumerate(row)) + b
              for row, b in zip(self.W2, self.b2)]
        y = [sigmoid(z) for z in z2]  # 网络最终输出 (0~1)
        return h, y

    # 反向传播 + 梯度下降更新参数
    def train_step(self, x, target, lr):
        h, y = self.forward(x)

        # 输出层误差：delta2 = (预测 - 真实) * sigmoid 导数
        delta2 = [(y[0] - target) * sigmoid_derivative(y[0])]

        # 隐藏层误差：delta1 = (W2^T @ delta2) * tanh 导数
        delta1 = []
        for j in range(len(self.W1)):
            err = sum(self.W2[k][j] * delta2[k] for k in range(len(delta2)))
            delta1.append(err * tanh_derivative(h[j]))

        # 更新 W2 和 b2
        for k in range(len(self.W2)):
            for j in range(len(self.W2[k])):
                self.W2[k][j] -= lr * delta2[k] * h[j]
            self.b2[k] -= lr * delta2[k]

        # 更新 W1 和 b1
        for j in range(len(self.W1)):
            for i in range(len(self.W1[j])):
                self.W1[j][i] -= lr * delta1[j] * x[i]
            self.b1[j] -= lr * delta1[j]

    # 均方误差损失（用于观察训练是否收敛）
    @staticmethod
    def loss(y, target):
        return (y[0] - target) ** 2


# ---------------------------------------------------------------------------
# 3. 训练与测试
# ---------------------------------------------------------------------------
def main():
    # XOR 数据集：两个输入，输出为 1 当且仅当两个输入不同
    data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]

    net = TwoLayerNet(input_size=2, hidden_size=4, output_size=1, seed=42)

    lr = 0.5          # 学习率
    epochs = 10000    # 训练轮数
    report_every = 2000

    print("=" * 50)
    print("训练一个 2-4-1 神经网络学习 XOR")
    print("=" * 50)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        # 每一轮把 4 个样本都过一遍（小批量=全量）
        for x, target in data:
            _, y = net.forward(x)
            total_loss += net.loss(y, target)
            net.train_step(x, target, lr)

        if epoch % report_every == 0 or epoch == 1:
            print(f"epoch {epoch:>6d} | 平均损失 = {total_loss / len(data):.6f}")

    print("\n训练结束，测试结果（阈值 0.5，输出 >= 0.5 视为 1）：")
    print("-" * 50)
    print("  输入    期望输出    实际输出    判定")
    ok = 0
    for x, target in data:
        _, y = net.forward(x)
        pred = 1 if y[0] >= 0.5 else 0
        flag = "✔" if pred == target else "✘"
        if pred == target:
            ok += 1
        print(f"  {x}        {target}        {y[0]:.4f}      {pred}  {flag}")

    print("-" * 50)
    print(f"准确率：{ok}/{len(data)}")
    print("\n结论：加上隐藏层后，神经网络成功学会了线性不可分的 XOR 问题。")


if __name__ == "__main__":
    main()
