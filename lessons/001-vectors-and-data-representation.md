# 第 1 课：向量与机器学习中的数据表示

## 1. 本课目标

完成本课后，你应该能够：

- 解释标量、向量和向量维度；
- 把一个真实样本表示成特征向量；
- 完成向量加法、数乘、点积、范数、距离和余弦相似度计算；
- 区分数学中的列向量与 NumPy 的一维数组；
- 检查向量运算的维度是否合法；
- 说明向量在回归、分类和 Embedding 中的用途。

配套内容：

- [练习题](../exercises/001-vectors.md)
- [参考答案](../solutions/001-vectors.md)
- [NumPy 示例](../code/lesson_001_vectors.py)

## 2. 为什么机器学习需要向量

机器学习模型不能直接处理“一个用户”“一张房屋信息表”或“一段文本”这样的抽象对象。我们需要先把对象转换成数字。

假设要预测一套房屋的价格，可以选择三个特征：

- 面积：80 平方米；
- 卧室数：2；
- 距离地铁站：0.6 千米。

这个样本可以表示成：

\[
x=
\begin{bmatrix}
80 \\
2 \\
0.6
\end{bmatrix}
\]

这就是一个三维向量。每个位置代表一个固定含义：

\[
x_1=80,\quad x_2=2,\quad x_3=0.6
\]

向量不仅是一串数字，它还携带两个关键信息：

1. 数字的顺序；
2. 每个位置对应的语义。

把特征顺序打乱，虽然数字没有变，样本含义已经改变。

## 3. 标量与向量

### 3.1 标量

标量是单个数值，通常使用普通小写字母表示：

\[
a=3,\quad \eta=0.01,\quad L=2.5
\]

机器学习中的标量示例：

- 学习率 `η`；
- 一个样本的损失 `L`；
- 一个分类概率 `p`；
- 一个偏置参数 `b`。

### 3.2 向量

向量是有顺序的一组数，通常使用粗体小写字母表示：

\[
\mathbf{x}=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_d
\end{bmatrix}
\in \mathbb{R}^{d}
\]

`d` 表示向量维度，`R^d` 表示由 `d` 个实数组成的空间。

例如：

\[
\mathbf{x}=
\begin{bmatrix}
2\\
-1\\
3
\end{bmatrix}
\in \mathbb{R}^{3}
\]

注意：向量的“维度”是元素数量，不是数组在程序中的轴数。

## 4. 列向量、行向量与 NumPy 一维数组

数学教材通常默认向量是列向量：

\[
\mathbf{x}\in\mathbb{R}^{d\times 1}
\]

它的转置是行向量：

\[
\mathbf{x}^{T}\in\mathbb{R}^{1\times d}
\]

在 NumPy 中：

```python
import numpy as np

x = np.array([2.0, -1.0, 3.0])
print(x.shape)  # (3,)
```

形状 `(3,)` 表示一维数组，它既不是严格的 `(3, 1)` 列向量，也不是 `(1, 3)` 行向量。

需要明确二维形状时：

```python
column = x.reshape(3, 1)  # (3, 1)
row = x.reshape(1, 3)     # (1, 3)
```

这是初学矩阵运算时最常见的错误来源之一。

## 5. 向量相等

两个向量相等，需要同时满足：

1. 维度相同；
2. 对应位置元素相同。

\[
\begin{bmatrix}1\\2\end{bmatrix}
=
\begin{bmatrix}1\\2\end{bmatrix}
\]

但：

\[
\begin{bmatrix}1\\2\end{bmatrix}
\ne
\begin{bmatrix}2\\1\end{bmatrix}
\]

元素相同但顺序不同，仍然是不同向量。

## 6. 向量加法

只有维度相同的向量才能相加：

\[
\mathbf{x}+\mathbf{y}
=
\begin{bmatrix}
x_1+y_1\\
x_2+y_2\\
\vdots\\
x_d+y_d
\end{bmatrix}
\]

例子：

\[
\begin{bmatrix}2\\-1\\3\end{bmatrix}
+
\begin{bmatrix}4\\2\\-2\end{bmatrix}
=
\begin{bmatrix}6\\1\\1\end{bmatrix}
\]

几何上，可以把向量看作位移。向量加法表示连续完成两次位移。

机器学习中的用途：

- 参数更新；
- 残差连接；
- Embedding 组合；
- 特征偏移。

## 7. 标量乘法

标量乘以向量，就是把每个元素都乘以该标量：

\[
c\mathbf{x}
=
\begin{bmatrix}
cx_1\\
cx_2\\
\vdots\\
cx_d
\end{bmatrix}
\]

例如：

\[
2
\begin{bmatrix}1\\-3\end{bmatrix}
=
\begin{bmatrix}2\\-6\end{bmatrix}
\]

几何上：

- `c > 1`：向量变长；
- `0 < c < 1`：向量变短；
- `c < 0`：方向翻转，同时按绝对值缩放；
- `c = 0`：得到零向量。

## 8. 点积

两个同维向量的点积是一个标量：

\[
\mathbf{x}^{T}\mathbf{y}
=
\sum_{i=1}^{d}x_i y_i
\]

例子：

\[
\mathbf{x}=
\begin{bmatrix}1\\2\\3\end{bmatrix},
\quad
\mathbf{y}=
\begin{bmatrix}4\\-1\\2\end{bmatrix}
\]

\[
\mathbf{x}^{T}\mathbf{y}
=1\times4+2\times(-1)+3\times2
=8
\]

### 8.1 点积的几何解释

\[
\mathbf{x}^{T}\mathbf{y}
=\|\mathbf{x}\|_2\|\mathbf{y}\|_2\cos\theta
\]

其中 `θ` 是两个向量之间的夹角。

因此：

- 点积大于 0：方向总体接近；
- 点积等于 0：两向量正交；
- 点积小于 0：方向总体相反。

### 8.2 在线性模型中的用途

线性模型通常写作：

\[
z=\mathbf{w}^{T}\mathbf{x}+b
\]

若 `w, x ∈ R^d`，那么：

- `w^T x` 是标量；
- `b` 是标量；
- `z` 也是标量。

可以把 `w` 理解为模型关注各个特征的方向和强度。

## 9. 范数

范数用于衡量向量大小。

### 9.1 L2 范数

\[
\|\mathbf{x}\|_2
=\sqrt{\sum_{i=1}^{d}x_i^2}
\]

例如：

\[
\left\|
\begin{bmatrix}3\\4\end{bmatrix}
\right\|_2
=\sqrt{3^2+4^2}=5
\]

它就是欧氏空间中的长度。

### 9.2 L1 范数

\[
\|\mathbf{x}\|_1
=\sum_{i=1}^{d}|x_i|
\]

对 `[3, -4]`：

\[
\|\mathbf{x}\|_1=3+4=7
\]

在机器学习中，L1 和 L2 范数常用于正则化，但它们为什么产生不同效果会在后续课程展开。

## 10. 距离

两个向量之间的欧氏距离：

\[
d(\mathbf{x},\mathbf{y})
=\|\mathbf{x}-\mathbf{y}\|_2
\]

例如：

\[
\mathbf{x}=
\begin{bmatrix}1\\2\end{bmatrix},
\quad
\mathbf{y}=
\begin{bmatrix}4\\6\end{bmatrix}
\]

\[
d(\mathbf{x},\mathbf{y})
=\sqrt{(1-4)^2+(2-6)^2}
=5
\]

距离会同时受到方向和长度影响。如果不同特征的量纲差别很大，距离可能被数值范围最大的特征支配。因此后续还需要学习特征缩放和标准化。

## 11. 余弦相似度

余弦相似度只关注方向：

\[
\operatorname{cosine}(\mathbf{x},\mathbf{y})
=
\frac{\mathbf{x}^{T}\mathbf{y}}
{\|\mathbf{x}\|_2\|\mathbf{y}\|_2}
\]

结果通常位于 `[-1, 1]`：

- 接近 `1`：方向相同；
- 接近 `0`：方向近似正交；
- 接近 `-1`：方向相反。

它常用于文本向量、Embedding 和检索系统。

注意：零向量的范数为 0，因此无法直接计算余弦相似度，代码必须显式处理。

## 12. 多个样本如何表示

一个样本是向量，多个样本通常堆叠成矩阵：

\[
X=
\begin{bmatrix}
---\mathbf{x}^{(1)T}---\\
---\mathbf{x}^{(2)T}---\\
\vdots\\
---\mathbf{x}^{(n)T}---
\end{bmatrix}
\in\mathbb{R}^{n\times d}
\]

其中：

- `n`：样本数；
- `d`：每个样本的特征数；
- 第 `i` 行：第 `i` 个样本；
- 第 `j` 列：第 `j` 个特征。

例如，3 套房屋、每套 3 个特征：

\[
X=
\begin{bmatrix}
80 & 2 & 0.6\\
105 & 3 & 1.2\\
60 & 1 & 0.3
\end{bmatrix}
\in\mathbb{R}^{3\times3}
\]

## 13. NumPy 中对应的运算

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, -1.0, 2.0])

vector_sum = x + y
scaled = 2.0 * x
dot_product = np.dot(x, y)
l2_norm = np.linalg.norm(x, ord=2)
distance = np.linalg.norm(x - y, ord=2)
cosine = dot_product / (
    np.linalg.norm(x) * np.linalg.norm(y)
)
```

需要区分：

```python
x * y   # 逐元素乘法
x @ y   # 一维数组时计算点积
```

后续矩阵课程中，`*` 和 `@` 的区别会更加重要。

## 14. 常见错误

### 错误 1：不同维度向量直接相加

`R^2` 和 `R^3` 中的向量不能逐位置对应相加。

### 错误 2：把逐元素乘法当成点积

```python
x * y
```

结果仍是向量；而：

```python
x @ y
```

对于两个一维数组，结果是标量。

### 错误 3：忽略特征顺序

`[面积, 卧室数, 距离]` 与 `[卧室数, 距离, 面积]` 不是同一种表示。

### 错误 4：混淆向量维度和数组形状

三维数学向量在 NumPy 中可能是 `(3,)`、`(3, 1)` 或 `(1, 3)`，它们参与广播和矩阵乘法时行为不同。

### 错误 5：对零向量计算余弦相似度

分母为 0，会得到无效结果。

### 错误 6：未经缩放直接比较距离

若一个特征范围是 `[0, 1]`，另一个是 `[0, 100000]`，欧氏距离几乎完全由第二个特征决定。

## 15. 本课检查

不看正文，尝试回答：

1. 向量的维度表示什么？
2. 点积的结果是标量还是向量？
3. 点积为什么能反映方向关系？
4. L1 与 L2 范数分别如何计算？
5. 欧氏距离与 L2 范数是什么关系？
6. 余弦相似度为什么不能直接用于零向量？
7. `np.array([1, 2, 3])` 的形状是什么？
8. `x * y` 与 `x @ y` 有什么区别？
9. 数据矩阵 `X ∈ R^(n×d)` 中，`n` 和 `d` 分别表示什么？
10. 为什么模型必须固定特征顺序？

## 16. 一句话总结

> 向量是机器学习表示单个样本、参数和特征方向的基本语言；点积、范数、距离和相似度则描述向量之间最基础的关系。