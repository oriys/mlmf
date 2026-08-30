# MLMF：机器学习数学基础

> Machine Learning Math Foundations

这是一个面向程序员的机器学习数学课程。目标不是先完整学习一遍数学专业课程，而是沿着机器学习模型所需的数学主线，逐步建立可推导、可实现、可诊断的数学能力。

## 当前内容

- 完整的机器学习数学知识地图；
- 六阶段学习路线；
- 可执行的 12 周课程计划；
- 五层掌握法与固定学习模板；
- 第 1 课完整讲义、分层练习和答案；
- 可运行的 NumPy 示例与错误诊断实验；
- 学习进度追踪表。

## 学习目标

完成本课程后，你应该能够：

- 看懂机器学习公式中的向量、矩阵、概率和优化符号；
- 独立推导线性回归、逻辑回归和两层神经网络；
- 理解 MSE、交叉熵、最大似然估计和正则化的来源；
- 理解梯度下降、反向传播、PCA、SVD 等核心方法；
- 使用 NumPy 验证公式，而不是只会调用框架 API；
- 识别维度错误、数值溢出、梯度异常和过拟合问题。

## 总体框架

```text
初等数学
├── 函数、指数、对数、求和
├── 线性代数：描述数据和模型
├── 微积分：描述参数变化
├── 概率论：描述不确定性
├── 数理统计：从样本推断总体
├── 最优化：寻找最优参数
├── 信息论：衡量信息和分布差异
└── 数值计算：保证实现稳定可靠
```

这些知识最终汇聚成机器学习的完整闭环：

```text
数据表示 → 构造模型 → 定义预测或概率 → 定义损失 → 计算梯度 → 优化参数 → 评估泛化
```

## 课程路线

| 阶段 | 主题 | 对应模型或能力 |
|---|---|---|
| Phase 0 | 数学语言与前置知识 | 函数、指数、对数、求和、数学符号 |
| Phase 1 | 线性代数 | 线性回归、最小二乘、PCA、Embedding |
| Phase 2 | 微积分与自动微分 | 梯度下降、计算图、反向传播 |
| Phase 3 | 概率论与数理统计 | MLE、MAP、贝叶斯、泛化 |
| Phase 4 | 最优化、信息论与数值计算 | SGD、Adam、交叉熵、稳定 Softmax |
| Phase 5 | 模型综合实践 | 逻辑回归、PCA、两层神经网络 |

核心导航：

- [学习路线](docs/01-roadmap.md)
- [学习方法](docs/02-study-method.md)
- [知识地图](docs/03-knowledge-map.md)
- [12 周课程计划](curriculum/12-week-plan.md)
- [学习进度](PROGRESS.md)

## 推荐学习方式

每个知识点都按照五层掌握法学习：

1. **直觉**：知道它解决什么问题；
2. **计算**：能够手工完成基础计算；
3. **推导**：能够从定义推导关键公式；
4. **实现**：能够使用 NumPy 实现并验证；
5. **诊断**：能够解释错误结果、训练异常和数值问题。

不要等到“数学全部学完”才开始模型。每学习一个数学概念，都应该立刻回答三个问题：

- 它在机器学习中解决什么问题？
- 它出现在什么模型或公式里？
- 我能否手算、推导并用代码验证？

## 第一课

从下面的顺序开始：

1. [讲义：向量与机器学习中的数据表示](lessons/001-vectors-and-data-representation.md)
2. [练习：向量与数据表示](exercises/001-vectors.md)
3. [配套 NumPy 代码](code/lesson001_vectors.py)
4. [练习答案](exercises/solutions/001-vectors-solutions.md)

建议先完成练习，再查看答案。

## 快速运行

```bash
git clone https://github.com/oriys/mlmf.git
cd mlmf

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/lesson001_vectors.py
```

Windows PowerShell 激活虚拟环境：

```powershell
.venv\Scripts\Activate.ps1
```

代码说明见：[配套代码使用指南](code/README.md)。

## 推荐学习顺序

先走通三条模型主线：

```text
线性回归
  ↓
逻辑回归
  ↓
两层神经网络
```

## 仓库结构

```text
mlmf/
├── README.md
├── PROGRESS.md          # 学习进度与复盘记录
├── requirements.txt    # Python 依赖
├── docs/               # 学习路线、方法和知识地图
├── curriculum/         # 12 周课程计划
├── lessons/            # 逐课学习内容
├── exercises/          # 手算、推导、编程、诊断题及答案
└── code/               # NumPy 实现与实验
```

## 学习原则

- 优先理解，不机械背公式；
- 每个公式都标注变量含义和维度；
- 先手算，再用代码验证；
- 先使用 NumPy，再使用自动微分框架；
- 数学与模型绑定学习；
- 每个阶段都应有可验证的产出。

## License

本项目采用 [MIT License](LICENSE)。