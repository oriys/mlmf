# 数学符号速查表

这份速查表收录机器学习基础阶段最常见的符号。阅读公式时，先确认符号的**语义、类型和形状**，再进行计算。

## 1. 数值、集合与形状

| 符号 | 含义 | 示例 |
|---|---|---|
| $\mathbb{R}$ | 实数集合 | $x\in\mathbb{R}$ |
| $\mathbb{R}^d$ | d 维实向量空间 | $x\in\mathbb{R}^d$ |
| $\mathbb{R}^{n\times d}$ | n 行 d 列实矩阵集合 | $X\in\mathbb{R}^{n\times d}$ |
| $\in$ | 属于 | $x\in S$ |
| $\subseteq$ | 子集 | $A\subseteq B$ |
| $\emptyset$ | 空集 | $A=\emptyset$ |

## 2. 常用变量

| 符号 | 常见含义 |
|---|---|
| $x$ | 单个输入样本或标量变量 |
| $y$ | 真实标签或目标值 |
| $\hat y$ | 模型预测值 |
| $X$ | 输入数据矩阵 |
| $W,w$ | 权重参数 |
| $b$ | 偏置参数 |
| $\theta$ | 全部模型参数 |
| $n$ | 样本数 |
| $d$ | 特征维度 |
| $k$ | 类别数或输出维度 |
| $L$ | 损失函数 |
| $\eta$ | 学习率 |
| $\lambda$ | 正则化强度 |

这些只是惯例，必须以公式上下文中的定义为准。

## 3. 标量、向量、矩阵和张量

本课程通常使用：

- 小写字母表示标量：$x$；
- 粗体小写字母或上下文明确的小写字母表示向量：$\mathbf{x}$；
- 大写字母表示矩阵：$X$；
- 高阶数组称为张量。

例如：

$$
x\in\mathbb{R},\qquad
\mathbf{x}\in\mathbb{R}^{d},\qquad
X\in\mathbb{R}^{n\times d}
$$

### 形状优先原则

看到：

$$
Y=XW
$$

应先写出：

$$
X\in\mathbb{R}^{n\times d},\quad
W\in\mathbb{R}^{d\times k},\quad
Y\in\mathbb{R}^{n\times k}
$$

只有左矩阵列数等于右矩阵行数，矩阵乘法才有定义。

## 4. 索引

| 符号 | 含义 |
|---|---|
| $x_i$ | 向量 x 的第 i 个元素 |
| $X_{ij}$ | 矩阵 X 的第 i 行第 j 列元素 |
| $x^{(i)}$ | 第 i 个样本，通常不是幂 |
| $W^{[l]}$ | 神经网络第 l 层参数 |
| $\theta_t$ | 第 t 次迭代时的参数 |

上标可能表示幂、样本编号、层编号或转置，必须根据上下文判断。

## 5. 向量与矩阵运算

| 符号 | 含义 |
|---|---|
| $x+y$ | 向量加法 |
| $\alpha x$ | 标量乘向量 |
| $x^Ty$ | 向量点积 |
| $XY$ | 矩阵乘法 |
| $X^T$ | 矩阵转置 |
| $X^{-1}$ | 矩阵逆，仅在可逆时存在 |
| $I$ | 单位矩阵 |
| $\operatorname{rank}(X)$ | 矩阵的秩 |
| $\operatorname{tr}(X)$ | 矩阵的迹 |
| $\det(X)$ | 行列式 |
| $x\odot y$ | 逐元素乘法 |

## 6. 范数、距离和相似度

### L1 范数

$$
\|x\|_1=\sum_{i=1}^{d}|x_i|
$$

### L2 范数

$$
\|x\|_2=\sqrt{\sum_{i=1}^{d}x_i^2}
$$

### 欧氏距离

$$
d(x,y)=\|x-y\|_2
$$

### 余弦相似度

$$
\operatorname{cos\_sim}(x,y)=
\frac{x^Ty}{\|x\|_2\|y\|_2}
$$

## 7. 求和、乘积和对数

$$
\sum_{i=1}^{n}x_i=x_1+x_2+\cdots+x_n
$$

$$
\prod_{i=1}^{n}p_i=p_1p_2\cdots p_n
$$

对数把乘积转换成求和：

$$
\log\prod_{i=1}^{n}p_i
=
\sum_{i=1}^{n}\log p_i
$$

这是最大似然估计通常使用对数似然的原因之一。

## 8. 函数与最优值

| 符号 | 含义 |
|---|---|
| $f(x)$ | 函数 f 在 x 处的值 |
| $\arg\min_x f(x)$ | 使 f 最小的 x |
| $\min_x f(x)$ | f 能达到的最小函数值 |
| $\arg\max_x f(x)$ | 使 f 最大的 x |

必须区分：

$$
x^*=\arg\min_x f(x)
$$

返回最优参数，而：

$$
f(x^*)=\min_x f(x)
$$

返回最小函数值。

## 9. 微积分符号

### 导数与偏导数

$$
\frac{df}{dx},\qquad f'(x),\qquad
\frac{\partial f}{\partial x_i}
$$

### 梯度

对 $f:\mathbb{R}^d\to\mathbb{R}$：

$$
\nabla_x f=
\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\vdots\\
\frac{\partial f}{\partial x_d}
\end{bmatrix}
$$

梯度形状与变量 x 相同。

### Jacobian 与 Hessian

对 $f:\mathbb{R}^d\to\mathbb{R}^k$：

$$
J_{ij}=\frac{\partial f_i}{\partial x_j}
$$

对标量函数 $f:\mathbb{R}^d\to\mathbb{R}$：

$$
H_{ij}=\frac{\partial^2 f}{\partial x_i\partial x_j}
$$

### 梯度下降

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t)
$$

## 10. 概率符号

| 符号 | 含义 |
|---|---|
| $P(A)$ | 事件 A 的概率 |
| $P(A\mid B)$ | 已知 B 时 A 的条件概率 |
| $p(x)$ | 概率质量函数或概率密度函数 |
| $p(x,y)$ | 联合分布 |
| $p(x\mid y)$ | 条件分布 |
| $X\sim\mathcal{N}(\mu,\sigma^2)$ | X 服从高斯分布 |
| $\mathbb{E}[X]$ | 期望 |
| $\operatorname{Var}(X)$ | 方差 |
| $\operatorname{Cov}(X,Y)$ | 协方差 |

贝叶斯公式：

$$
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}
$$

## 11. 统计与机器学习符号

### 最大似然估计

$$
\hat\theta_{\mathrm{MLE}}
=
\arg\max_\theta
\sum_{i=1}^{n}\log p(x^{(i)}\mid\theta)
$$

### 最大后验估计

$$
\hat\theta_{\mathrm{MAP}}
=
\arg\max_\theta p(\theta\mid D)
$$

### 经验风险最小化

$$
\hat\theta=
\arg\min_\theta
\frac{1}{n}\sum_{i=1}^{n}
\ell\bigl(f_\theta(x^{(i)}),y^{(i)}\bigr)
$$

## 12. 常见损失函数

### 均方误差

$$
L_{\mathrm{MSE}}
=
\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2
$$

### 二分类交叉熵

$$
L
=
-\frac{1}{n}\sum_{i=1}^{n}
\left[
y_i\log p_i+(1-y_i)\log(1-p_i)
\right]
$$

### 多分类交叉熵

$$
L
=
-\frac{1}{n}\sum_{i=1}^{n}
\sum_{c=1}^{k}y_{ic}\log p_{ic}
$$

## 13. 矩阵分解

特征值与特征向量：

$$
Av=\lambda v
$$

奇异值分解：

$$
X=U\Sigma V^T
$$

## 14. 信息论符号

$$
H(P)=-\sum_x P(x)\log P(x)
$$

$$
H(P,Q)=-\sum_x P(x)\log Q(x)
$$

$$
D_{\mathrm{KL}}(P\|Q)
=
\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

## 15. 阅读公式的固定流程

每次看到新公式，按顺序检查：

1. 每个符号表示什么；
2. 每个变量是标量、向量、矩阵还是随机变量；
3. 每个变量的形状是什么；
4. 等号两边形状是否一致；
5. 这是定义、恒等式、近似还是优化目标；
6. 公式依赖哪些假设；
7. 如何用一个最小数值例子验证。

不要跳过形状检查。机器学习中的许多理解错误和代码错误，本质上都是变量语义或维度没有确认清楚。