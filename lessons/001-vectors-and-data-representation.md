# 第 1 课：向量与机器学习中的数据表示

## 本课目标

完成本课后，你应该能够：

- 区分标量、向量和矩阵；
- 使用向量表示一个样本；
- 使用矩阵表示一批样本；
- 正确计算向量加法、数乘和点积；
- 解释点积为什么是线性模型的核心；
- 为机器学习公式中的变量标注 shape；
- 使用 NumPy 验证手算结果；
- 识别行向量、列向量和广播导致的常见错误。

配套练习：[`exercises/001-vectors.md`](../exercises/001-vectors.md)

配套代码：[`code/lesson001_vectors.py`](../code/lesson001_vectors.py)

---

## 1. 为什么机器学习从向量开始

现实世界中的对象不能直接交给模型计算。模型真正接收的是数字。

例如，一个房屋样本可以包含：

- 面积：80 平方米；
- 卧室数：2；
- 房龄：10 年；
- 距市中心距离：5 千米。

可以把它写成向量：

$$
x=
\begin{bmatrix}
80\\
2\\
10\\
5
\end{bmatrix}
$$

这个向量不是单纯的一列数字。每个位置都有明确语义：

| 下标 | 特征 | 数值 |
|---|---|---:|
| $x_1$ | 面积 | 80 |
| $x_2$ | 卧室数 | 2 |
| $x_3$ | 房龄 | 10 |
| $x_4$ | 距市中心距离 | 5 |

因此，向量是机器学习中“一个样本的结构化数值表示”。

---

## 2. 标量、向量和矩阵

### 2.1 标量

标量是单个数字，例如：

$$
a=3.5
$$

机器学习中的标量包括：

- 一个标签；
- 一个学习率；
- 一个损失值；
- 一个权重；
- 一个概率。

通常写成：

$$
a\in\mathbb{R}
$$

意思是 $a$ 是一个实数。

### 2.2 向量

向量是有顺序的一组数字：

$$
x=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_d
\end{bmatrix}
\in\mathbb{R}^{d}
$$

其中 $d$ 是向量的维度，也就是元素数量。

如果一个样本有 4 个特征，那么：

$$
x\in\mathbb{R}^{4}
$$

### 2.3 矩阵

矩阵可以表示一批样本。假设有 $n$ 个样本，每个样本有 $d$ 个特征：

$$
X\in\mathbb{R}^{n\times d}
$$

通常约定：

- 每一行是一个样本；
- 每一列是一个特征。

例如，3 个房屋样本、每个样本 4 个特征：

$$
X=
\begin{bmatrix}
80 & 2 & 10 & 5\\
120 & 3 & 5 & 8\\
60 & 1 & 20 & 3
\end{bmatrix}
\in\mathbb{R}^{3\times4}
$$

这里：

- $n=3$；
- $d=4$。

---

## 3. shape 是公式的类型系统

程序设计中，类型系统可以阻止不合法操作。在线性代数中，shape 起到类似作用。

看到公式时，先不要计算，先写 shape。

例如：

$$
X\in\mathbb{R}^{n\times d},
\qquad
w\in\mathbb{R}^{d},
\qquad
b\in\mathbb{R}
$$

线性模型：

$$
\hat y=Xw+b
$$

shape 推导：

$$
(n\times d)(d\times1)=n\times1
$$

因此：

$$
\hat y\in\mathbb{R}^{n}
$$

如果把 $w$ 错写成 $n$ 维向量，矩阵乘法就不成立。

### 3.1 一条实用规则

矩阵乘法：

$$
A_{m\times n}B_{n\times p}=C_{m\times p}
$$

中间维度必须相同，结果保留外侧维度。

可以记成：

```text
(m × n) @ (n × p) → (m × p)
```

---

## 4. 行向量与列向量

数学中通常把特征向量写成列向量：

$$
x=
\begin{bmatrix}
x_1\\x_2\\x_3
\end{bmatrix}
\in\mathbb{R}^{3\times1}
$$

它的转置是行向量：

$$
x^T=
\begin{bmatrix}
x_1 & x_2 & x_3
\end{bmatrix}
\in\mathbb{R}^{1\times3}
$$

但在 NumPy 中，一维数组：

```python
x = np.array([1.0, 2.0, 3.0])
```

它的 shape 是：

```text
(3,)
```

它既不是严格的 `(3, 1)` 列矩阵，也不是 `(1, 3)` 行矩阵。这是很多维度错误的来源。

需要二维列向量时应显式写：

```python
x_column = np.array([[1.0], [2.0], [3.0]])
# shape: (3, 1)
```

需要二维行向量时：

```python
x_row = np.array([[1.0, 2.0, 3.0]])
# shape: (1, 3)
```

---

## 5. 向量的基本运算

设：

$$
x=
\begin{bmatrix}1\\2\\3\end{bmatrix},
\qquad
y=
\begin{bmatrix}4\\5\\6\end{bmatrix}
$$

### 5.1 向量加法

$$
x+y=
\begin{bmatrix}
1+4\\2+5\\3+6
\end{bmatrix}
=
\begin{bmatrix}
5\\7\\9
\end{bmatrix}
$$

只有维度相同的向量才能直接相加。

### 5.2 标量乘法

$$
2x=
\begin{bmatrix}
2\\4\\6
\end{bmatrix}
$$

它会同时缩放向量的所有分量。

### 5.3 逐元素乘法

$$
x\odot y=
\begin{bmatrix}
1\cdot4\\2\cdot5\\3\cdot6
\end{bmatrix}
=
\begin{bmatrix}
4\\10\\18
\end{bmatrix}
$$

在 NumPy 中对应：

```python
x * y
```

逐元素乘法与点积不是同一个操作。

---

## 6. 点积

两个同维向量的点积定义为：

$$
x^Ty=\sum_{i=1}^{d}x_i y_i
$$

对前面的例子：

$$
x^Ty=1\cdot4+2\cdot5+3\cdot6=32
$$

点积的结果是一个标量。

### 6.1 点积的三个视角

#### 视角一：对应元素相乘后求和

这是计算视角：

```text
multiply → sum
```

#### 视角二：加权求和

假设：

$$
x=
\begin{bmatrix}
80\\2\\10\\5
\end{bmatrix},
\qquad
w=
\begin{bmatrix}
0.5\\20\\-1\\-3
\end{bmatrix}
$$

那么：

$$
w^Tx
=0.5\times80+20\times2-1\times10-3\times5
=55
$$

每个特征乘以对应权重，然后相加。这正是线性模型的核心。

#### 视角三：方向相似程度

点积还满足：

$$
x^Ty=\lVert x\rVert_2\lVert y\rVert_2\cos\theta
$$

因此点积同时受两个因素影响：

- 向量长度；
- 向量方向是否接近。

当两个非零向量：

- 同方向时，点积为正且较大；
- 垂直时，点积为 0；
- 反方向时，点积为负。

---

## 7. 点积如何构成线性模型

单个样本的线性模型：

$$
\hat y=w^Tx+b
$$

其中：

- $x\in\mathbb{R}^{d}$：输入特征；
- $w\in\mathbb{R}^{d}$：每个特征的权重；
- $b\in\mathbb{R}$：偏置；
- $\hat y\in\mathbb{R}$：预测结果。

展开后：

$$
\hat y=w_1x_1+w_2x_2+\cdots+w_dx_d+b
$$

这说明线性模型做了三件事：

1. 每个特征乘以一个权重；
2. 把加权结果求和；
3. 加上偏置。

### 7.1 一批样本

把 $n$ 个样本堆叠成矩阵：

$$
X=
\begin{bmatrix}
- & x_1^T & -\\
- & x_2^T & -\\
  & \vdots & \\
- & x_n^T & -
\end{bmatrix}
\in\mathbb{R}^{n\times d}
$$

则全部预测可以一次完成：

$$
\hat y=Xw+b
$$

这叫向量化。它避免手工循环每个样本，也更接近底层高性能线性代数实现。

---

## 8. 矩阵乘法不是逐元素乘法

设：

$$
A=
\begin{bmatrix}
1&2\\3&4
\end{bmatrix},
\qquad
B=
\begin{bmatrix}
5&6\\7&8
\end{bmatrix}
$$

逐元素乘法：

$$
A\odot B=
\begin{bmatrix}
5&12\\21&32
\end{bmatrix}
$$

矩阵乘法：

$$
AB=
\begin{bmatrix}
1\cdot5+2\cdot7 & 1\cdot6+2\cdot8\\
3\cdot5+4\cdot7 & 3\cdot6+4\cdot8
\end{bmatrix}
=
\begin{bmatrix}
19&22\\43&50
\end{bmatrix}
$$

在 NumPy 中：

```python
A * B   # 逐元素乘法
A @ B   # 矩阵乘法
```

机器学习代码中混淆 `*` 和 `@` 是常见错误。

---

## 9. NumPy 最小实验

```python
import numpy as np

X = np.array([
    [80.0, 2.0, 10.0, 5.0],
    [120.0, 3.0, 5.0, 8.0],
    [60.0, 1.0, 20.0, 3.0],
])

w = np.array([0.5, 20.0, -1.0, -3.0])
b = 10.0

predictions = X @ w + b

assert X.shape == (3, 4)
assert w.shape == (4,)
assert predictions.shape == (3,)

print(predictions)
```

手算第一个样本：

$$
0.5\times80+20\times2-1\times10-3\times5+10=65
$$

程序第一个输出也应为 `65`。

---

## 10. 广播：方便，但可能掩盖错误

NumPy 可以把标量 $b$ 自动加到向量的每个元素：

```python
predictions = X @ w + b
```

这里：

- `X @ w` 的 shape 是 `(3,)`；
- `b` 是标量；
- NumPy 把 `b` 加到三个预测值上。

但广播也可能让错误代码“可以运行，却不是你想要的结果”。

例如：

```python
column = np.ones((3, 1))
row = np.ones((3,))
result = column + row
```

结果 shape 是 `(3, 3)`，不是 `(3, 1)`。

因此，机器学习实现中应主动写 shape 断言。

---

## 11. 常见错误

### 错误 1：不写变量维度

只看符号，很容易在转置和矩阵乘法上出错。

改进：每个新公式先写：

```text
变量含义 + shape
```

### 错误 2：混淆样本数和特征数

推荐约定：

$$
X\in\mathbb{R}^{n\times d}
$$

其中：

- $n$：样本数；
- $d$：特征数。

### 错误 3：把逐元素乘法当成矩阵乘法

```python
X * w
```

得到的是逐元素广播结果；

```python
X @ w
```

才是每个样本与权重的点积。

### 错误 4：认为 NumPy 一维数组有明确行列方向

`shape == (d,)` 不等于 `(d, 1)`。

### 错误 5：代码能运行就认为数学正确

广播可能让错误 shape 继续执行。必须检查结果是否符合预期维度和语义。

### 错误 6：只记点积公式，不理解它在模型中的意义

点积是“特征乘权重后求和”。理解这一点，才能自然理解线性回归、逻辑回归、神经网络线性层和注意力得分。

---

## 12. 本课知识与后续模型的连接

### 线性回归

$$
\hat y=Xw+b
$$

### 逻辑回归

$$
p(y=1\mid x)=\sigma(w^Tx+b)
$$

### 神经网络线性层

$$
Z=XW+b
$$

### 注意力机制

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

它们虽然复杂程度不同，但底层都大量依赖向量点积和矩阵乘法。

---

## 13. 本课验收清单

不看本课内容，尝试完成：

- [ ] 用自己的话解释标量、向量和矩阵；
- [ ] 把一个现实样本表示成向量；
- [ ] 把一批样本表示成矩阵；
- [ ] 手算两个三维向量的点积；
- [ ] 解释点积的加权求和意义；
- [ ] 判断 `(5, 3) @ (3, 2)` 的结果 shape；
- [ ] 区分 NumPy 中 `*` 和 `@`；
- [ ] 解释 `(3,)` 与 `(3, 1)` 的区别；
- [ ] 为 $Xw+b$ 标注每个变量的维度；
- [ ] 独立运行并修改配套代码。

完成后进入配套练习：[`001-vectors.md`](../exercises/001-vectors.md)。