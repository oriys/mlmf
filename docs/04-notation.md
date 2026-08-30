# 数学符号与维度规范

本项目统一使用这份符号约定。不同教材可能采用不同写法，阅读外部资料时应关注定义，而不是只认字母。

## 字体约定

| 类型 | 示例 | 含义 |
|---|---|---|
| 普通小写字母 | \(x,y,n\) | 标量或索引 |
| 粗体小写字母 | \(\mathbf{x},\mathbf{w}\) | 向量 |
| 大写字母 | \(X,W,A\) | 矩阵 |
| 花体大写字母 | \(\mathcal{D},\mathcal{L}\) | 数据集、集合或函数族 |
| 希腊字母 | \(\theta,\mu,\sigma\) | 参数、均值、标准差等 |

在纯文本或代码里，粗体通常被省略。因此 `x` 的真实类型必须结合定义和形状判断。

## 集合与数域

| 符号 | 含义 |
|---|---|
| \(x\in A\) | `x` 属于集合 `A` |
| \(A\subseteq B\) | `A` 是 `B` 的子集 |
| \(\mathbb{R}\) | 实数集合 |
| \(\mathbb{R}^d\) | `d` 维实向量空间 |
| \(\mathbb{R}^{n\times d}\) | `n` 行 `d` 列的实矩阵集合 |
| \(\varnothing\) | 空集 |

## 标量、向量与矩阵

### 标量

\[
x\in\mathbb{R}
\]

表示 `x` 是一个实数。

### 向量

\[
\mathbf{x}\in\mathbb{R}^{d}
\]

表示 `x` 是长度为 `d` 的向量。默认把向量视为列向量：

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
X\in\mathbb{R}^{n\times d}
\]

通常表示有 `n` 条样本、每条样本有 `d` 个特征：

\[
X=
\begin{bmatrix}
---\mathbf{x}_1^T---\\
---\mathbf{x}_2^T---\\
\vdots\\
---\mathbf{x}_n^T---
\end{bmatrix}
\]

本项目默认：

- 行表示样本；
- 列表示特征；
- `n` 表示样本数；
- `d` 表示输入特征数；
- `k` 或 `c` 表示输出维度或类别数。

## 下标与上标

| 写法 | 常见含义 |
|---|---|
| \(x_i\) | 向量第 `i` 个元素，或第 `i` 个样本 |
| \(X_{ij}\) | 矩阵第 `i` 行、第 `j` 列元素 |
| \(\mathbf{x}^{(i)}\) | 第 `i` 个样本；上标不是幂 |
| \(w_t\) | 第 `t` 次迭代的参数 |
| \(X^T\) | 矩阵转置 |
| \(X^{-1}\) | 矩阵逆；不是逐元素倒数 |

使用 `(i)` 表示样本编号，可以减少与向量分量下标的冲突。

## 常用运算

### 向量点积

\[
\mathbf{x}^T\mathbf{w}
=
\sum_{j=1}^{d}x_jw_j
\]

输入形状：

\[
\mathbf{x},\mathbf{w}\in\mathbb{R}^{d}
\]

输出是标量。

### 矩阵乘法

若：

\[
A\in\mathbb{R}^{m\times n},\quad
B\in\mathbb{R}^{n\times p}
\]

则：

\[
AB\in\mathbb{R}^{m\times p}
\]

内部维度必须相同，外部维度决定结果形状。

### Hadamard 逐元素乘法

\[
A\odot B
\]

表示对应元素相乘。它和矩阵乘法 `AB` 不同。

### 范数

L1 范数：

\[
\|\mathbf{x}\|_1=\sum_i|x_i|
\]

L2 范数：

\[
\|\mathbf{x}\|_2=\sqrt{\sum_i x_i^2}
\]

平方 L2 范数：

\[
\|\mathbf{x}\|_2^2=\sum_i x_i^2
\]

注意平方 L2 范数没有根号，求导更方便。

## 求和、乘积与均值

求和：

\[
\sum_{i=1}^{n}x_i=x_1+x_2+\cdots+x_n
\]

乘积：

\[
\prod_{i=1}^{n}p_i=p_1p_2\cdots p_n
\]

样本均值：

\[
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
\]

独立样本的似然通常是概率乘积，取对数后变为对数概率之和：

\[
\log\prod_{i=1}^{n}p_i
=
\sum_{i=1}^{n}\log p_i
\]

## 函数与最优化

函数映射：

\[
f:\mathbb{R}^{d}\rightarrow\mathbb{R}^{k}
\]

表示输入是 `d` 维向量，输出是 `k` 维向量。

最小值：

\[
\min_{\theta}L(\theta)
\]

表示损失函数能够取得的最小数值。

最小值对应的参数：

\[
\theta^*=\arg\min_{\theta}L(\theta)
\]

`min` 返回函数值，`argmin` 返回使函数最小的参数。

## 微积分符号

一元导数：

\[
\frac{df}{dx}
\]

偏导数：

\[
\frac{\partial f}{\partial x_i}
\]

梯度：

\[
\nabla_{\mathbf{x}}f=
\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\vdots\\
\frac{\partial f}{\partial x_d}
\end{bmatrix}
\]

若 `f` 输出标量、`x` 是 `d` 维向量，则梯度形状和 `x` 相同。

Jacobian：

若：

\[
\mathbf{f}:\mathbb{R}^{d}\rightarrow\mathbb{R}^{k}
\]

则 Jacobian 常写为：

\[
J_{\mathbf{f}}\in\mathbb{R}^{k\times d}
\]

Hessian：

对标量函数 \(f:\mathbb{R}^{d}\rightarrow\mathbb{R}\)：

\[
H_f=\nabla^2 f\in\mathbb{R}^{d\times d}
\]

## 概率符号

| 符号 | 含义 |
|---|---|
| \(P(A)\) | 事件 `A` 的概率 |
| \(P(A\mid B)\) | 已知 `B` 时 `A` 的条件概率 |
| \(p(x)\) | 随机变量在 `x` 处的概率质量或密度 |
| \(p(x,y)\) | 联合分布 |
| \(p(x\mid y)\) | 条件分布 |
| \(X\sim\mathcal{N}(\mu,\sigma^2)\) | `X` 服从高斯分布 |
| \(\mathbb{E}[X]\) | 期望 |
| \(\mathrm{Var}(X)\) | 方差 |
| \(\mathrm{Cov}(X,Y)\) | 协方差 |

注意：概率中的大写 `X` 常表示随机变量，小写 `x` 表示随机变量的一次具体取值。

## 数据集和损失函数

监督学习数据集：

\[
\mathcal{D}=\{(\mathbf{x}^{(i)},y^{(i)})\}_{i=1}^{n}
\]

单样本损失：

\[
\ell(f_\theta(\mathbf{x}^{(i)}),y^{(i)})
\]

经验风险：

\[
L(\theta)=\frac{1}{n}\sum_{i=1}^{n}
\ell(f_\theta(\mathbf{x}^{(i)}),y^{(i)})
\]

带正则化的目标：

\[
J(\theta)=L(\theta)+\lambda R(\theta)
\]

其中：

- `L` 衡量拟合误差；
- `R` 限制模型复杂度；
- `λ` 控制二者权衡。

## 维度检查模板

每次推导或写代码前，先写：

```text
X: (n, d)     n 个样本，d 个特征
W: (d, k)     从 d 维映射到 k 维
b: (k,)       每个输出维度一个偏置
Y: (n, k)     每条样本得到 k 维输出
```

于是：

```text
X @ W          (n, d) @ (d, k) -> (n, k)
X @ W + b      (n, k) + (k,)    -> (n, k)，b 按行广播
```

## 常见符号陷阱

1. `x²` 是平方，`x^(i)` 通常是第 `i` 个样本；
2. `AB` 是矩阵乘法，`A ⊙ B` 是逐元素乘法；
3. `P(A|B)` 一般不等于 `P(B|A)`；
4. 概率密度可以大于 1，但某个区间的概率不能大于 1；
5. `∇f` 的方向约定和分子布局可能因教材不同而变化；
6. 向量是行还是列必须从上下文确认；
7. `X⁻¹y` 不应在代码中机械实现为 `inv(X) @ y`，通常应直接解线性方程；
8. 相同字母在不同章节可能代表不同概念，始终以当前定义为准。