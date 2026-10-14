# -*- coding: utf-8 -*-
"""
perceptron.py —— 从零实现的感知机（Perceptron）学习算法

感知机是最早的神经网络模型（1957 年 Rosenblatt 提出），
只能解决"线性可分"的问题（能用一条直线/一个平面分开的数据）。

本脚本演示三件事：
  1. 感知机成功学会 AND（与门）——线性可分
  2. 感知机成功学会 OR（或门） ——线性可分
  3. 感知机学不会 XOR（异或）——线性不可分（这正是后来神经网络
     需要加隐藏层、发展成多层感知机的原因）

运行方式：  python perceptron.py
"""

import random


class Perceptron:
    """单层感知机：输入向量 x -> 加权求和 -> 阈值判断 -> 输出 0/1"""

    def __init__(self, n_inputs, seed=None):
        # 用独立的 Random 实例，避免影响全局随机状态
        rng = random.Random(seed)
        # 权重随机初始化，偏置初始为 0
        self.weights = [rng.uniform(-0.5, 0.5) for _ in range(n_inputs)]
        self.bias = 0.0

    # 前向计算：加权和 + 偏置，大于 0 输出 1，否则输出 0
    def predict(self, x):
        s = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return 1 if s > 0 else 0

    # 训练一个样本：预测错了才更新权重
    # 更新规则：w = w + lr * (真实值 - 预测值) * x
    def train_step(self, x, target, lr):
        pred = self.predict(x)
        error = target - pred  # 只有错误时才非 0
        if error != 0:
            for i in range(len(self.weights)):
                self.weights[i] += lr * error * x[i]
            self.bias += lr * error

    # 完整训练：把所有样本反复过很多轮，直到全部正确或达到上限
    def train(self, data, lr=0.1, max_epochs=200):
        for epoch in range(1, max_epochs + 1):
            # 先过一遍所有样本，更新权重
            for x, target in data:
                self.train_step(x, target, lr)
            # 本轮结束时，对全部样本统一评估一次，全对才算收敛
            # （不能边更新边判断——同轮内前面学对的样本可能又被后面的样本改错）
            errors = sum(1 for x, target in data if self.predict(x) != target)
            if errors == 0:
                return epoch
        return None  # 达到上限仍未收敛（说明数据线性不可分）


def run_logic_gate(name, data):
    """在某个逻辑门上训练感知机并输出结果"""
    p = Perceptron(n_inputs=2, seed=7)
    epochs = p.train(data, lr=0.1, max_epochs=500)

    print(f"\n【{name}】")
    if epochs is None:
        print("  训练 500 轮仍未收敛 —— 数据线性不可分，感知机学不会！")
    else:
        print(f"  训练 {epochs} 轮后收敛 ✔")
        ok = 0
        for x, target in data:
            pred = p.predict(x)
            flag = "✔" if pred == target else "✘"
            if pred == target:
                ok += 1
            print(f"  输入 {x} -> 期望 {target}，实际 {pred} {flag}")
        print(f"  权重 = {[round(w, 3) for w in p.weights]}，偏置 = {round(p.bias, 3)}")
        print(f"  准确率：{ok}/{len(data)}")


def main():
    # AND：只有两个输入都是 1 时输出 1
    and_data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    # OR：只要有一个输入是 1 就输出 1
    or_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]
    # XOR：两个输入不同才输出 1
    xor_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

    print("=" * 55)
    print("单层感知机学习逻辑门：AND / OR 应该能学会，XOR 学不会")
    print("=" * 55)

    run_logic_gate("AND（与门）", and_data)
    run_logic_gate("OR（或门）", or_data)
    run_logic_gate("XOR（异或）", xor_data)

    print("\n" + "=" * 55)
    print("结论：")
    print("  AND/OR 在平面上可以用一条直线分开（线性可分），感知机可收敛；")
    print("  XOR 用一条直线永远分不开（线性不可分），所以需要多层网络。")
    print("  试试运行 nn_xor.py，看两层神经网络如何解决这个问题！")
    print("=" * 55)


if __name__ == "__main__":
    main()
