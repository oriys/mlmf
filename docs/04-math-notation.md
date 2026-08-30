# 数学符号速查

机器学习学习中的很多困难，并不是概念本身，而是不熟悉符号。本文统一仓库内使用的记号。

## 1. 数与集合

| 符号 | 含义 | 示例 |
|---|---|---|
| $\mathbb{R}$ | 实数集合 | $x\in\mathbb{R}$ |
| $\mathbb{R}^d$ | $d$ 维实向量空间 | $x\in\mathbb{R}^d$ |
| $\mathbb{R}^{n\times d}$ | $n$ 行 $d$ 列实矩阵集合 | $X\in\mathbb{R}^{n\times d}$ |
| $\in$ | 属于 | $x\in\mathbb{R}$ |
| $\notin$ | 不属于 | $x\notin A$ |
| $\subseteq$ | 子集 | $A\subseteq B$ |
| $\forall$ | 对所有 | $\forall i\in\{1,\ldots,n\}$ |
| $\exists$ | 存在 | $\exists x$ |

## 2. 标量、向量、矩阵和张量

仓库默认使用以下约定：

- 普通小写字母表示标量：$x,y,b,\eta$；
- 粗体小写字母表示向量：$\mathbf{x},\mathbf{w}$；
- 粗体大写字母表示矩阵：$\mathbf{X},\mathbf{W}$；
- 花体大写字母表示集合或数据集：$\mathcal{D},\mathcal{X}$；
- 希腊字母常表示模型参数或统计量：$\theta,\mu,\sigma$。

由于 GitHub Markdown 的字体渲染限制，部分文档会直接写 $x$、$w$、$X$，并通过上下文和形状声明区分。

### 形状声明

$$
X\in\mathbb{R}^{n\times d}
$$

表示：

- $n$：样本数量；
- $d$：每个样本的特征数量；
- 第 $i$ 行 $x_i^T$：第 $i$ 个样本；
- 第 $j$ 列：第 $j$ 个特征在所有样本上的取值。

### 常见模型形状

$$
X\in\mathbb{R}^{n\times d},\qquad
w\in\mathbb{R}^{d},\qquad
b\in\mathbb{R},\qquad
y\in\mathbb{R}^{n}
$$

线性预测：

$$
\hat y=Xw+b\mathbf{1}
$$

结果形状为：

$$
\hat y\in\mathbb{R}^{n}
$$

## 3. 下标与上标

| 写法 | 常见含义 |
|---|---|
| $x_i$ | 第 $i$ 个元素，或第 $i$ 个样本 |
| $x_{ij}$ | 矩阵第 $i$ 行第 $j$ 列元素 |
| $x^{(i)}$ | 第 $i$ 个样本；用于避免和向量分量混淆 |
| $w_t$ | 第 $t$ 次迭代时的参数 |
| $X^T$ | 矩阵转置，不表示幂 |
| $A^{-1}$ | 矩阵逆 |
| $x^2$ | 标量平方 |

文档会优先使用 $x^{(i)}$ 表示第 $i$ 个样本，使用 $x_j$ 表示一个向量的第 $j$ 个特征。

## 4. 求和、乘积和均值

### 求和

$$
\sum_{i=1}^{n}x_i=x_1+x_2+\cdots+x_n
$$

### 均值

$$
\bar x=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

### 连乘

$$
\prod_{i=1}^{n}p_i=p_1p_2\cdots p_n
$$

对数可以把连乘转换为求和：

$$
\log\prod_{i=1}^{n}p_i
=
\sum_{i=1}^{n}\log p_i
$$

这就是机器学习中经常优化对数似然而不是原始似然的原因之一。

## 5. 函数和映射

$$
f:\mathbb{R}^{d}\rightarrow\mathbb{R}^{k}
$$

表示函数 $f$ 接收一个 $d$ 维向量，输出一个 $k$ 维向量。

模型常写为：

$$
\hat y=f_{\theta}(x)
$$

其中：

- $x$：输入；
- $\theta$：模型参数；
- $f_{\theta}$：由参数 $\theta$ 决定的函数；
- $\hat y$：预测结果。

## 6. 常见线性代数符号

| 符号 | 含义 |
|---|---|
| $x^Ty$ | 向量点积 |
| $AB$ | 矩阵乘法 |
| $A^T$ | 转置 |
| $A^{-1}$ | 逆矩阵 |
| $I$ | 单位矩阵 |
| $\lVert x\rVert_1$ | L1 范数 |
| $\lVert x\rVert_2$ | L2 范数 |
| $\operatorname{rank}(A)$ | 矩阵的秩 |
| $\operatorname{tr}(A)$ | 矩阵的迹 |
| $\det(A)$ | 行列式 |
| $x\perp y$ | $x$ 与 $y$ 正交 |
| $\lambda$ | 常用作特征值或正则化系数 |

### 点积

$$
x^Ty=\sum_{j=1}^{d}x_jy_j
$$

### L2 范数

$$
\lVert x\rVert_2=\sqrt{\sum_{j=1}^{d}x_j^2}
$$

### 外积

当 $x\in\mathbb{R}^{d}$、$y\in\mathbb{R}^{k}$ 时：

$$
xy^T\in\mathbb{R}^{d\times k}
$$

## 7. 微积分符号

| 符号 | 含义 |
|---|---|
| $\frac{df}{dx}$ | 一元函数导数 |
| $\frac{\partial f}{\partial x_i}$ | 对变量 $x_i$ 的偏导数 |
| $\nabla_x f$ | $f$ 对向量 $x$ 的梯度 |
| $J_f$ | Jacobian 矩阵 |
| $H_f$ 或 $\nabla^2f$ | Hessian 矩阵 |
| $\Delta x$ | 变量变化量 |

若 $f:\mathbb{R}^{d}\to\mathbb{R}$，则梯度为列向量：

$$
\nabla_x f=
\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\vdots\\
\frac{\partial f}{\partial x_d}
\end{bmatrix}
\in\mathbb{R}^{d}
$$

## 8. 概率与统计符号

| 符号 | 含义 |
|---|---|
| $P(A)$ | 事件 $A$ 的概率 |
| $P(A\mid B)$ | 在 $B$ 已发生时 $A$ 的条件概率 |
| $p(x)$ | 随机变量的概率质量或密度 |
| $X\sim\mathcal{N}(\mu,\sigma^2)$ | $X$ 服从高斯分布 |
| $\mathbb{E}[X]$ | 期望 |
| $\operatorname{Var}(X)$ | 方差 |
| $\operatorname{Cov}(X,Y)$ | 协方差 |
| $X\perp Y$ | 随机变量独立；需结合上下文判断 |
| $\hat\theta$ | 参数估计值 |

注意：$P(A\mid B)$ 与 $P(B\mid A)$ 通常不同。

## 9. 优化符号

### 最小值与使函数最小的参数

$$
\min_{\theta}L(\theta)
$$

表示最小化损失值。

$$
\theta^*=\arg\min_{\theta}L(\theta)
$$

表示取得最小损失时的参数。

### 梯度下降

$$
\theta_{t+1}=\theta_t-\eta\nabla_{\theta}L(\theta_t)
$$

其中：

- $t$：迭代编号；
- $\eta$：学习率；
- $\nabla_{\theta}L$：损失对参数的梯度。

## 10. 数据与机器学习约定

| 符号 | 默认含义 |
|---|---|
| $n$ | 样本数 |
| $d$ | 特征数 |
| $k$ 或 $C$ | 类别数或输出维度 |
| $X$ | 数据矩阵 |
| $x^{(i)}$ | 第 $i$ 个输入样本 |
| $y^{(i)}$ | 第 $i$ 个真实标签 |
| $\hat y^{(i)}$ | 第 $i$ 个预测结果 |
| $w,b$ | 线性模型参数 |
| $\theta$ | 全部模型参数的统称 |
| $L$ | 总损失函数 |
| $\ell$ | 单样本损失函数 |
| $\eta$ | 学习率 |
| $\lambda$ | 正则化强度 |

经验风险通常写为：

$$
L(\theta)=\frac{1}{n}\sum_{i=1}^{n}
\ell\left(f_{\theta}(x^{(i)}),y^{(i)}\right)
$$

## 11. 阅读公式的固定流程

看到一个新公式时，依次回答：

1. 每个符号代表标量、向量还是矩阵？
2. 每个变量的形状是什么？
3. 运算是否合法？
4. 公式的输入和输出是什么？
5. 它在描述模型、概率、损失还是参数更新？
6. 能否用一个二维小例子手算验证？

先完成维度分析，再进行代数推导，可以避免大量理解和实现错误。