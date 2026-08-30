# MLMF：机器学习数学基础

[![tests](https://github.com/oriys/mlmf/actions/workflows/tests.yml/badge.svg)](https://github.com/oriys/mlmf/actions/workflows/tests.yml)

> Machine Learning Math Foundations

这是一个面向程序员的机器学习数学课程。目标不是先完整学习一遍数学专业课程，而是沿着机器学习模型所需的主线，逐步建立**能解释、能计算、能推导、能实现、能诊断**的数学能力。

## 立即开始

1. 阅读[学习路线](docs/01-roadmap.md)；
2. 了解[五层学习方法](docs/02-study-method.md)；
3. 查看[完整课程目录](curriculum/README.md)；
4. 开始[第 1 课：向量与数据表示](lessons/001-vectors-and-data-representation.md)；
5. 独立完成[第 1 课练习](exercises/001-vectors-and-data-representation.md)；
6. 完成后再核对[第 1 课参考答案](solutions/001-vectors-and-data-representation.md)；
7. 在 [PROGRESS.md](PROGRESS.md) 记录进度和错误。

查阅资料：

- [机器学习数学知识地图](docs/03-knowledge-map.md)
- [数学符号与 Shape 规范](docs/04-notation.md)
- [贡献与课程编写规范](CONTRIBUTING.md)

## 学习目标

完成课程后，你应该能够：

- 看懂机器学习公式中的向量、矩阵、导数、概率和优化符号；
- 为输入、参数、中间结果和梯度准确标注 Shape；
- 独立推导线性回归、逻辑回归和两层神经网络；
- 理解 MSE、交叉熵、最大似然估计和正则化的来源；
- 理解梯度下降、反向传播、PCA、SVD 等核心方法；
- 使用 NumPy 验证公式，而不是只会调用框架 API；
- 识别维度错误、广播错误、数值溢出、梯度异常和过拟合问题。

## 数学基础框架

```text
初等数学
├── 函数、指数、对数、求和
│
├── 线性代数 ─────────────── 数据与模型的表示
│   └── 向量、矩阵、投影、特征值、SVD
│
├── 微积分 ───────────────── 参数变化与梯度
│   └── 偏导数、链式法则、Jacobian、Hessian
│
├── 概率论 ───────────────── 不确定性的表示
│   └── 条件概率、分布、期望、方差
│
├── 数理统计 ─────────────── 从样本推断总体
│   └── MLE、MAP、偏差、方差、泛化
│
├── 最优化 ───────────────── 寻找更好的参数
│   └── GD、SGD、Momentum、Adam、凸优化
│
├── 信息论 ───────────────── 衡量信息与分布差异
│   └── 熵、交叉熵、KL 散度
│
└── 数值计算 ─────────────── 让公式稳定可靠地运行
    └── 浮点误差、条件数、Log-Sum-Exp、稳定 Softmax
```

这些知识最终汇聚为一个完整闭环：

```text
数据表示
→ 构造模型
→ 定义预测或概率
→ 定义损失函数
→ 计算梯度
→ 优化参数
→ 评估泛化
→ 诊断失败
```

## 课程路线

| 阶段 | 主题 | 对应模型或能力 |
|---|---|---|
| Phase 0 | 数学语言与前置知识 | 函数、指数、对数、求和和数学符号 |
| Phase 1 | 线性代数 | 线性回归、最小二乘、PCA、Embedding |
| Phase 2 | 微积分与自动微分 | 梯度下降、计算图和反向传播 |
| Phase 3 | 概率论与数理统计 | MLE、MAP、贝叶斯和泛化 |
| Phase 4 | 最优化、信息论与数值计算 | SGD、Adam、交叉熵和稳定 Softmax |
| Phase 5 | 模型综合实践 | 逻辑回归、K-Means、PCA、两层网络和 Attention |

完整的 36 节核心课与阶段项目见[课程目录](curriculum/README.md)。

## 五层掌握法

每个知识点都沿着同一条路径学习：

```text
直觉 → 计算 → 推导 → 实现 → 诊断
```

1. **直觉**：知道它解决什么问题；
2. **计算**：能够完成小规模手算；
3. **推导**：能够从定义得到关键公式；
4. **实现**：能够用 NumPy 实现并验证；
5. **诊断**：能够解释维度、数值、优化和泛化问题。

“看过”不算完成。数学、代码与模型行为真正连接起来，才算掌握。

## 三条模型主线

优先走通：

```text
线性回归
  ↓
逻辑回归
  ↓
两层神经网络
```

这三条主线会依次串起：

```text
线性代数
+ 微积分
+ 概率统计
+ 信息论
+ 最优化
+ 数值计算
```

不要等数学全部学完才开始模型。每学一个概念，都回答：

- 它解决了什么问题？
- 它出现在什么模型里？
- 我能否手算、推导并用代码验证？

## 本地运行

需要 Python 3.11 及以上版本。

```bash
git clone https://github.com/oriys/mlmf.git
cd mlmf

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python code/lesson_001_vectors.py
python -m pytest -q
```

Windows PowerShell 激活虚拟环境：

```powershell
.venv\Scripts\Activate.ps1
```

## 仓库结构

```text
mlmf/
├── README.md
├── PROGRESS.md                    # 学习进度和失败实验记录
├── CONTRIBUTING.md                # 课程与代码编写规范
├── docs/                          # 路线、方法、知识地图、符号规范
├── curriculum/                    # 36 节核心课的完整索引
├── lessons/                       # 逐课正文
├── exercises/                     # 计算、推导、编程和诊断练习
├── solutions/                     # 独立完成练习后再查看的参考答案
├── code/                          # 可直接运行的 NumPy 演示程序
├── mlmf/                          # 经过输入校验的参考实现
├── tests/                         # 正常、边界和错误输入测试
├── .github/workflows/tests.yml    # 持续集成
├── requirements.txt
└── LICENSE
```

## 当前内容

首个可运行学习闭环已经完成：

```text
第 1 课正文
→ 四类练习
→ 参考答案
→ NumPy 向量实现
→ 正常与错误输入测试
→ GitHub Actions 自动校验
```

第 1 课覆盖：

- 特征向量与数据契约；
- 标量、向量、矩阵和张量；
- 向量加法、数乘和点积；
- L1、L2 范数、距离与余弦相似度；
- 单样本和 Batch 线性得分；
- NumPy 一维向量陷阱；
- Shape、广播、单位和特征顺序错误。

## 学习原则

- 优先理解，不机械背公式；
- 每个公式都标注变量含义和 Shape；
- 先手算，再使用代码验证；
- 先使用 NumPy，再依赖自动微分框架；
- 数学概念必须与模型绑定；
- 手写梯度必须进行梯度检查；
- 记录失败实验，而不是只保留成功结果；
- 每个阶段都以可验证产出验收。

## License

本项目采用 [MIT License](LICENSE)。