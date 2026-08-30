# 数学符号与维度速查

机器学习公式难读，往往不是概念本身复杂，而是变量、形状和运算规则没有先说清楚。本页作为全课程统一的符号约定。

## 1. 数与集合

| 符号 | 含义 |
|---|---|
| \(\mathbb{N}\) | 自然数集合 |
| \(\mathbb{Z}\) | 整数集合 |
| \(\mathbb{R}\) | 实数集合 |
| \(a\in A\) | 元素 \(a\) 属于集合 \(A\) |
| \(A\subseteq B\) | \(A\) 是 \(B\) 的子集 |
| \([a,b]\) | 包含端点的闭区间 |
| \((a,b)\) | 不包含端点的开区间 |

## 2. 标量、向量、矩阵与张量

本课程默认使用下面的字体约定：

- 普通小写字母表示标量：\(x,y,\alpha,\lambda\)；
- 粗体小写字母表示向量：\(\mathbf{x},\mathbf{w}\)；
- 粗体大写字母表示矩阵：\(\mathbf{X},\mathbf{W}\)；
- 花体字母常表示集合或数据集：\(\mathcal{D}\)。

### 标量

\[
x\in\mathbb{R}
\]

表示 `x` 是一个实数。

### 向量

\[
\mathbf{x}\in\mathbb{R}^{d}
\]

表示 `x` 是一个包含 `d` 个元素的向量。

除非特别说明，本课程把单个向量视为列向量：

\[
\mathbf{x}=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_d
\end{bmatrix}
\]

### 矩阵

\[
\mathbf{X}\in\mathbb{R}^{n\times d}
\]

表示 `X` 有 `n` 行、`d` 列。在数据集中通常约定：

- 每一行是一条样本；
- 每一列是一个特征；
- `n` 是样本数量；
- `d` 是特征数量。

### 张量

张量是向量和矩阵向更高维度的推广。例如图像 Batch：

\[
\mathbf{X}\in\mathbb{R}^{N\times C\times H\times W}
\]

其中 `N` 是样本数，`C` 是通道数，`H` 和 `W` 是高度和宽度。

## 3. 索引约定

| 符号 | 含义 |
|---|---|
| \(x_i\) | 向量的第 `i` 个元素 |
| \(X_{ij}\) | 矩阵第 `i` 行、第 `j` 列元素 |
| \(\mathbf{x}^{(i)}\) | 第 `i` 条样本 |
| \(y^{(i)}\) | 第 `i` 条样本的标签 |
| \(w_j\) | 第 `j` 个模型参数 |

上标括号 \((i)\) 表示样本编号，不表示幂。

## 4. 常见向量运算

### 向量加法

\[
\mathbf{x}+\mathbf{y}
\]

要求两个向量形状相同，结果形状不变。

### 数乘

\[
\alpha\mathbf{x}
\]

表示向量每个元素都乘以标量 \(\alpha\)。

### 点积

\[
\mathbf{x}^{T}\mathbf{y}
=
\sum_{i=1}^{d}x_i y_i
\]

如果 \(\mathbf{x},\mathbf{y}\in\mathbb{R}^{d}\)，点积结果是标量。

### Hadamard 逐元素乘法

\[
\mathbf{x}\odot\mathbf{y}
\]

与点积不同，它返回长度为 `d` 的向量。

### 外积

\[
\mathbf{x}\mathbf{y}^{T}
\]

如果 \(\mathbf{x}\in\mathbb{R}^{m}\)、\(\mathbf{y}\in\mathbb{R}^{n}\)，结果是 \(m\times n\) 矩阵。

## 5. 矩阵运算与维度检查

### 矩阵乘法

\[
\mathbf{A}\mathbf{B}
\]

若：

\[
\mathbf{A}\in\mathbb{R}^{m\times n},\qquad
\mathbf{B}\in\mathbb{R}^{n\times k}
\]

则：

\[
\mathbf{A}\mathbf{B}\in\mathbb{R}^{m\times k}
\]

记忆方式：内部维度必须相同，外部维度决定结果。

```text
(m × n) @ (n × k) = (m × k)
```

### 转置

\[
\mathbf{A}^{T}\in\mathbb{R}^{n\times m}
\]

常用性质：

\[
(\mathbf{A}\mathbf{B})^{T}=\mathbf{B}^{T}\mathbf{A}^{T}
\]

注意转置后乘法顺序反转。

### 逆矩阵

\[
\mathbf{A}^{-1}\mathbf{A}=\mathbf{I}
\]

只有方阵且可逆时才存在逆矩阵。工程实现中通常优先解线性方程组，而不是显式计算逆矩阵。

## 6. 范数、距离与相似度

### L1 范数

\[
\|\mathbf{x}\|_1=\sum_i |x_i|
\]

### L2 范数

\[
\|\mathbf{x}\|_2=\sqrt{\sum_i x_i^2}
\]

### 欧氏距离

\[
d(\mathbf{x},\mathbf{y})=\|\mathbf{x}-\mathbf{y}\|_2
\]

### 余弦相似度

\[
\cos(\mathbf{x},\mathbf{y})
=
\frac{\mathbf{x}^{T}\mathbf{y}}
{\|\mathbf{x}\|_2\|\mathbf{y}\|_2}
\]

当任意一个向量为零向量时，分母为零，余弦相似度没有定义。

## 7. 函数与微积分

### 函数

\[
f:\mathbb{R}^{d}\rightarrow\mathbb{R}
\]

表示函数输入一个 `d` 维向量，输出一个标量。

### 一元导数

\[
\frac{df}{dx},\qquad f'(x)
\]

### 偏导数

\[
\frac{\partial f}{\partial x_i}
\]

表示其他变量暂时保持不变时，函数相对 `x_i` 的变化率。

### 梯度

若 \(f:\mathbb{R}^{d}\to\mathbb{R}\)，则：

\[
\nabla_{\mathbf{x}}f
=
\begin{bmatrix}
\partial f/\partial x_1\\
\vdots\\
\partial f/\partial x_d
\end{bmatrix}
\in\mathbb{R}^{d}
\]

梯度的形状与输入向量相同。

### Jacobian

若 \(\mathbf{f}:\mathbb{R}^{d}\to\mathbb{R}^{k}\)，则 Jacobian：

\[
\mathbf{J}\in\mathbb{R}^{k\times d}
\]

### Hessian

若 \(f:\mathbb{R}^{d}\to\mathbb{R}\)，则 Hessian：

\[
\mathbf{H}\in\mathbb{R}^{d\times d}
\]

它由所有二阶偏导数组成，用于描述局部曲率。

## 8. 概率与统计

| 符号 | 含义 |
|---|---|
| \(P(A)\) | 事件 `A` 的概率 |
| \(P(A\mid B)\) | 在 `B` 已发生条件下 `A` 的概率 |
| \(p(x)\) | 离散概率质量函数或连续概率密度的简写 |
| \(X\sim\mathcal{N}(\mu,\sigma^2)\) | `X` 服从高斯分布 |
| \(\mathbb{E}[X]\) | 期望 |
| \(\operatorname{Var}(X)\) | 方差 |
| \(\operatorname{Cov}(X,Y)\) | 协方差 |
| \(X\perp Y\) | `X` 与 `Y` 独立 |

注意：概率密度值本身可以大于 1，但密度在区间上的积分才是概率。

## 9. 求和、期望与 Batch

### 求和

\[
\sum_{i=1}^{n}x_i
\]

### 平均值

\[
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
\]

### 经验风险

机器学习中常见平均损失：

\[
L(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\ell\left(f_{\theta}(\mathbf{x}^{(i)}),y^{(i)}\right)
\]

其中：

- \(\theta\) 是全部模型参数；
- \(f_{\theta}\) 是参数化模型；
- \(\ell\) 是单样本损失函数；
- \(L\) 是整个数据集上的平均损失。

## 10. 优化符号

### 最小值与最小化参数

\[
\min_{\theta}L(\theta)
\]

表示最小的函数值。

\[
\theta^{*}=\arg\min_{\theta}L(\theta)
\]

表示取得最小值时的参数。

`min` 返回值，`argmin` 返回参数，两者不要混淆。

### 梯度下降

\[
\theta_{t+1}
=
\theta_t-\eta\nabla_{\theta}L(\theta_t)
\]

其中 \(\eta\) 是学习率，`t` 是迭代编号。

## 11. 维度检查示例

线性模型：

\[
\hat{\mathbf{y}}=\mathbf{X}\mathbf{w}+b
\]

给定：

\[
\mathbf{X}\in\mathbb{R}^{n\times d},\qquad
\mathbf{w}\in\mathbb{R}^{d},\qquad
b\in\mathbb{R}
\]

那么：

\[
\mathbf{X}\mathbf{w}\in\mathbb{R}^{n}
\]

标量 `b` 会通过广播加到每个样本，因此：

\[
\hat{\mathbf{y}}\in\mathbb{R}^{n}
\]

反过来，若把权重错误地写成 \(\mathbf{w}\in\mathbb{R}^{n}\)，则 `Xw` 在矩阵乘法开始前就能判断不合法。

## 12. 阅读任何公式时的固定步骤

1. 找出输出变量；
2. 列出所有输入变量及其语义；
3. 标注每个变量的类型和形状；
4. 从左到右检查每一步运算是否合法；
5. 把公式翻译成自然语言；
6. 找出背后的假设；
7. 再讨论如何求解或实现。

先做维度分析，通常能消除一半以上的公式阅读困难。