# -*- coding: utf-8 -*-
"""
markov_text.py —— 用马尔可夫链（Markov Chain）生成文本

原理（说人话版）：
  统计一段文字里"每个词后面通常跟着什么词"，建一张"下一个词"的转移表。
  生成时从某个词出发，随机挑选它后面常跟的词接上，不断重复，
  就能拼出"看起来像人话"的新文本。这就是很多早期聊天机器人
  和"短信生成器"的基础，也是大语言模型的雏形思路（n-gram 模型）。

本脚本使用内置中文语料，不依赖任何第三方库。

运行方式：  python markov_text.py
"""

import random


def build_transition_table(text, order=1):
    """
    根据语料构建马尔可夫转移表。

    order=1：只看前 1 个词决定下一个词（一阶马尔可夫）
    order=2：看前 2 个词决定下一个词（二阶，生成的句子连贯性更好）
    """
    # 按空白切词，去掉多余空格
    words = text.split()
    table = {}  # 键：前序词元组，值：后一个词的列表

    for i in range(len(words) - order):
        key = tuple(words[i:i + order])   # 前序词
        nxt = words[i + order]            # 后一个词
        table.setdefault(key, []).append(nxt)
    return table


def generate_text(table, length=80, start=None):
    """
    从转移表随机生成一段文本。

    length：生成多少个词
    start：起始词（不指定则随机选一个）
    """
    if not table:
        return "(语料太短，无法生成)"

    order = len(next(iter(table.keys())))  # 从表里推断阶数

    if start is None:
        key = random.choice(list(table.keys()))
    else:
        key = start
        # 如果指定了起始词，把它拼成同样长度的元组
        if isinstance(start, str):
            key = tuple(start.split()[:order])

    words = list(key)

    for _ in range(length):
        # 如果当前前序词不在表里，随机挑一个新的起点继续
        if key not in table:
            key = random.choice(list(table.keys()))
            words.extend(key)
            continue
        # 随机选一个"下一个词"；如果存在其他选择，尽量避免和当前词重复，
        # 减少"词语原地打转"的情况，让生成结果更连贯
        options = table[key]
        if len(options) > 1:
            others = [o for o in options if o != words[-1]]
            nxt = random.choice(others if others else options)
        else:
            nxt = options[0]
        words.append(nxt)
        # 窗口向后滑动一格：丢掉最前面的词，接上刚生成的词
        key = tuple(list(key[1:]) + [nxt])

    return "".join(words)


def main():
    # 内置中文语料（一段介绍人工智能的小短文）
    corpus = """
    人工智能是研究如何让机器像人一样思考和行动的学科 机器学习是人工智能
    的核心方法 它让计算机从数据中自动学习规律 而不需要人工一条条编写规则
    深度学习是机器学习的一个分支 它使用多层神经网络 在图像识别 语音识别
    自然语言处理等领域取得了巨大成功 大语言模型是近年最火热的方向 它能
    理解并生成自然语言 帮助人们写文章 翻译 编程 甚至聊天 神经网络的基本
    单元是神经元 大量神经元通过权重连接组成网络 通过大量数据训练 网络能
    自动调整权重 从而学会各种任务 生成式人工智能正在改变人们的工作方式
    越来越多的人开始使用人工智能工具提高效率 未来 人工智能会像电力一样
    成为基础设施 深刻影响每个人的生活
    """.strip()

    print("=" * 60)
    print("马尔可夫链文本生成（内置 AI 科普语料）")
    print("=" * 60)

    # 一阶：只看前 1 个词
    table1 = build_transition_table(corpus, order=1)
    random.seed(1)
    print("\n【一阶马尔可夫链生成的文本】")
    print(generate_text(table1, length=60))

    # 二阶：看前 2 个词，连贯性更好
    table2 = build_transition_table(corpus, order=2)
    random.seed(1)
    print("\n【二阶马尔可夫链生成的文本】")
    print(generate_text(table2, length=60))

    print("\n" + "=" * 60)
    print("小提示：把语料换成你喜欢的文章（比如歌词、小说），")
    print("生成的文本风格就会跟着变。试试 order=3 会更连贯哦！")
    print("=" * 60)


if __name__ == "__main__":
    main()
