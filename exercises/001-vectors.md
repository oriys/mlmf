# 第 1 课练习：向量与数据表示

> 建议先手算，再使用 NumPy 验证。不要一开始查看答案。

答案：[`solutions/001-vectors-solutions.md`](solutions/001-vectors-solutions.md)

---

## A. 概念理解

### A1

分别判断下面对象是标量、向量还是矩阵，并写出 shape：

1. 一名学生的数学成绩 `92`；
2. 一名学生三门课的成绩 `[92, 85, 88]`；
3. 30 名学生三门课的成绩表；
4. 一次训练结束后的总损失；
5. 一个包含 768 个数字的文本 Embedding。

### A2

用自己的话解释：为什么一个向量不仅是一组数字，还必须包含“每个位置的语义”？

### A3

解释下面两个 shape 的区别：

```text
(4,)
(4, 1)
```

### A4

矩阵 $X\in\mathbb{R}^{100\times20}$ 通常表示什么？分别说明 `100` 和 `20` 的含义。

### A5

为什么机器学习实现中应该主动写 shape 断言，即使代码当前能够运行？

### A6

说明逐元素乘法与点积的区别。两者的输出类型通常分别是什么？

---

## B. 手工计算

### B1：向量加法和数乘

给定：

$$
x=\begin{bmatrix}2\\-1\\3\end{bmatrix},
\qquad
y=\begin{bmatrix}4\\5\\-2\end{bmatrix}
$$

计算：

1. $x+y$；
2. $x-y$；
3. $3x$；
4. $-2y$。

### B2：逐元素乘法

计算：

$$
x\odot y
$$

### B3：点积

计算：

$$
x^Ty
$$

并判断结果是标量还是向量。

### B4：线性模型

一个房屋样本：

$$
x=\begin{bmatrix}80\\2\\10\\5\end{bmatrix}
$$

权重和偏置：

$$
w=\begin{bmatrix}0.5\\20\\-1\\-3\end{bmatrix},
\qquad b=10
$$

计算：

$$
\hat y=w^Tx+b
$$

写出每个特征对最终结果的贡献。

### B5：矩阵乘法

给定：

$$
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},
\qquad
B=\begin{bmatrix}1&2\\0&1\\2&0\end{bmatrix}
$$

1. 判断 $AB$ 是否可计算；
2. 写出结果 shape；
3. 手工计算 $AB$。

### B6：批量预测

给定：

$$
X=
\begin{bmatrix}
1&2\\
3&4\\
5&6
\end{bmatrix},
\qquad
w=
\begin{bmatrix}
2\\-1
\end{bmatrix},
\qquad b=3
$$

计算：

$$
\hat y=Xw+b
$$

---

## C. 维度推理

不进行数值计算，只判断是否合法并写出结果 shape。

### C1

$$
A\in\mathbb{R}^{5\times3},
\quad B\in\mathbb{R}^{3\times2},
\quad AB=?
$$

### C2

$$
A\in\mathbb{R}^{5\times3},
\quad B\in\mathbb{R}^{5\times2},
\quad AB=?
$$

### C3

$$
X\in\mathbb{R}^{128\times64},
\quad w\in\mathbb{R}^{64},
\quad Xw=?
$$

### C4

$$
X\in\mathbb{R}^{128\times64},
\quad W\in\mathbb{R}^{64\times10},
\quad XW=?
$$

### C5

$$
Q\in\mathbb{R}^{32\times128},
\quad K\in\mathbb{R}^{64\times128},
\quad QK^T=?
$$

### C6

如果：

$$
X\in\mathbb{R}^{n\times d},
\quad W\in\mathbb{R}^{d\times h},
\quad b\in\mathbb{R}^{h}
$$

说明为什么：

$$
XW+b\in\mathbb{R}^{n\times h}
$$

### C7

一个二分类数据集有 500 个样本、12 个特征。分别为 $X$、$w$、$y$ 和批量预测 $\hat y$ 选择合理 shape。

### C8

下面代码中 `result` 的 shape 是什么？为什么它可能不是开发者想要的结果？

```python
column = np.ones((3, 1))
row = np.ones((3,))
result = column + row
```

---

## D. NumPy 编程

### D1：表示数据

创建一个包含 4 个样本、每个样本 3 个特征的数据矩阵 `X`，并使用断言检查：

```python
assert X.shape == (4, 3)
```

### D2：手算验证

创建：

```python
x = np.array([2.0, -1.0, 3.0])
y = np.array([4.0, 5.0, -2.0])
```

分别计算：

- 向量加法；
- 逐元素乘法；
- 点积。

把程序结果与 B 部分手算结果比较。

### D3：批量线性预测

编写函数：

```python
def linear_predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    ...
```

要求：

- 检查 `X` 为二维数组；
- 检查 `w` 为一维数组；
- 检查 `X.shape[1] == w.shape[0]`；
- 返回 shape 为 `(n,)` 的预测结果。

### D4：行列向量实验

分别创建 shape 为：

```text
(3,)
(1, 3)
(3, 1)
```

的三个数组。对它们进行转置，并记录每个结果的 shape。解释为什么一维数组转置后 shape 没有变化。

### D5：矩阵乘法与逐元素乘法

创建两个 shape 都为 `(2, 2)` 的矩阵，分别执行：

```python
A * B
A @ B
```

解释两个结果为什么不同。

### D6：向量化与循环

对同一批数据，分别使用：

1. Python `for` 循环逐样本计算 $w^Tx+b$；
2. `X @ w + b` 一次性计算。

验证结果相同，并说明向量化写法的优势。

---

## E. 错误诊断

### E1：错误的矩阵方向

```python
X = np.ones((100, 20))
w = np.ones((100,))
predictions = X @ w
```

回答：

1. 这段代码为什么失败？
2. `w` 应该是什么 shape？
3. 这个错误在语义上混淆了什么？

### E2：错误使用逐元素乘法

```python
X = np.ones((3, 4))
w = np.arange(4)
predictions = X * w
```

代码能够运行，但 `predictions.shape == (3, 4)`。

回答：

1. 为什么能够运行？
2. 为什么它不是三个样本的线性预测？
3. 正确写法是什么？

### E3：广播产生意外矩阵

```python
prediction = np.ones((8, 1))
target = np.ones((8,))
error = prediction - target
```

1. `error` 的 shape 是什么？
2. 为什么可能导致错误的损失计算？
3. 给出两种修复方式。

### E4：错误的转置假设

```python
x = np.array([1.0, 2.0, 3.0])
assert x.T.shape == (1, 3)
```

为什么断言失败？如何得到真正的行向量？

### E5：缺少输入校验

某个预测函数在输入特征数量错误时仍然通过广播产生结果。请设计至少三个 shape 断言，尽早暴露问题。

---

## F. 综合任务

设计一个简单的“候选人匹配分数”模型。

每个候选人包含四个特征：

1. 工作年限；
2. 技术面试得分；
3. 沟通得分；
4. 与岗位距离。

完成：

1. 设计三个候选人的数据矩阵 $X$；
2. 说明每一列的语义和单位；
3. 设计权重 $w$ 与偏置 $b$；
4. 使用 $Xw+b$ 计算分数；
5. 解释每个权重正负的含义；
6. 分析“不同特征单位差异很大”可能带来的问题；
7. 尝试调整一个权重，解释排序如何变化。

---

## 提交检查

- [ ] 所有手算题写出了中间过程；
- [ ] 所有矩阵公式标注了 shape；
- [ ] 编程题包含断言；
- [ ] 能解释代码为什么正确，而不只是给出输出；
- [ ] 完成至少三个错误诊断题；
- [ ] 最后再对照答案订正。