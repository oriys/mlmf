# 第 1 课练习答案：向量与数据表示

建议先独立完成练习，再对照答案。答案的重点不是最终数字，而是计算过程、shape 推理和错误原因。

---

## A. 概念理解

### A1

1. 数学成绩 `92`：标量，shape 可视为 `()`；
2. 三门课成绩：向量，shape 为 `(3,)`；
3. 30 名学生三门课成绩表：矩阵，shape 为 `(30, 3)`；
4. 一次训练的总损失：标量，shape 为 `()`；
5. 768 维文本 Embedding：向量，shape 为 `(768,)`。

### A2

向量中的位置有固定含义。`[80, 2, 10, 5]` 只有在知道四个位置分别代表面积、卧室数、房龄和距离时，才能被正确解释。任意交换两个位置，数学上仍是四维向量，但业务语义已经改变。

### A3

- `(4,)`：NumPy 一维数组，没有明确的行或列方向；
- `(4, 1)`：二维列向量，4 行 1 列。

一维数组转置后仍是 `(4,)`，而 `(4, 1)` 转置后是 `(1, 4)`。

### A4

`X ∈ R^(100×20)` 通常表示：

- 100 行：100 个样本；
- 20 列：每个样本有 20 个特征。

### A5

NumPy 广播可能让维度错误的代码继续运行。shape 断言可以在数据刚偏离设计时立即失败，而不是等到损失值、训练结果或线上预测异常后再排查。

### A6

- 逐元素乘法：对应位置相乘，两个同维向量通常输出同维向量；
- 点积：对应位置相乘后求和，两个同维向量输出标量。

---

## B. 手工计算

给定：

$$
x=\begin{bmatrix}2\\-1\\3\end{bmatrix},
\qquad
y=\begin{bmatrix}4\\5\\-2\end{bmatrix}
$$

### B1

$$
x+y=
\begin{bmatrix}
2+4\\-1+5\\3-2
\end{bmatrix}
=
\begin{bmatrix}6\\4\\1\end{bmatrix}
$$

$$
x-y=
\begin{bmatrix}
2-4\\-1-5\\3-(-2)
\end{bmatrix}
=
\begin{bmatrix}-2\\-6\\5\end{bmatrix}
$$

$$
3x=\begin{bmatrix}6\\-3\\9\end{bmatrix}
$$

$$
-2y=\begin{bmatrix}-8\\-10\\4\end{bmatrix}
$$

### B2

$$
x\odot y=
\begin{bmatrix}
2\cdot4\\(-1)\cdot5\\3\cdot(-2)
\end{bmatrix}
=
\begin{bmatrix}8\\-5\\-6\end{bmatrix}
$$

### B3

$$
x^Ty=2\cdot4+(-1)\cdot5+3\cdot(-2)
=8-5-6=-3
$$

结果是标量。

### B4

各特征贡献：

| 特征 | 计算 | 贡献 |
|---|---:|---:|
| 面积 | `0.5 × 80` | 40 |
| 卧室数 | `20 × 2` | 40 |
| 房龄 | `-1 × 10` | -10 |
| 距离 | `-3 × 5` | -15 |
| 偏置 | `b` | 10 |

因此：

$$
\hat y=40+40-10-15+10=65
$$

### B5

$A$ 的 shape 是 `(2, 3)`，$B$ 的 shape 是 `(3, 2)`。中间维度相同，因此可以相乘，结果 shape 是 `(2, 2)`。

$$
AB=
\begin{bmatrix}
1\cdot1+2\cdot0+3\cdot2 & 1\cdot2+2\cdot1+3\cdot0\\
4\cdot1+5\cdot0+6\cdot2 & 4\cdot2+5\cdot1+6\cdot0
\end{bmatrix}
=
\begin{bmatrix}
7&4\\16&13
\end{bmatrix}
$$

### B6

$$
Xw=
\begin{bmatrix}
1\cdot2+2\cdot(-1)\\
3\cdot2+4\cdot(-1)\\
5\cdot2+6\cdot(-1)
\end{bmatrix}
=
\begin{bmatrix}0\\2\\4\end{bmatrix}
$$

加上偏置：

$$
\hat y=Xw+b=
\begin{bmatrix}3\\5\\7\end{bmatrix}
$$

---

## C. 维度推理

### C1

```text
(5 × 3) @ (3 × 2) → (5 × 2)
```

合法，结果 shape 为 `(5, 2)`。

### C2

```text
(5 × 3) @ (5 × 2)
```

不合法，中间维度 `3` 和 `5` 不相同。

### C3

```text
(128 × 64) @ (64,) → (128,)
```

结果 shape 为 `(128,)`。

### C4

```text
(128 × 64) @ (64 × 10) → (128 × 10)
```

结果 shape 为 `(128, 10)`。

### C5

$K^T$ 的 shape 是 `(128, 64)`：

```text
(32 × 128) @ (128 × 64) → (32 × 64)
```

结果 shape 为 `(32, 64)`。

### C6

先计算：

```text
(n × d) @ (d × h) → (n × h)
```

$b$ 的 shape 是 `(h,)`，NumPy 会沿样本维广播，相当于给每一行加上相同的偏置向量，因此结果仍为 `(n, h)`。

### C7

合理 shape：

- `X`: `(500, 12)`；
- `w`: `(12,)`；
- `y`: `(500,)`；
- `y_hat`: `(500,)`。

### C8

`column` 是 `(3, 1)`，`row` 是 `(3,)`。广播时 `(3,)` 被视为 `(1, 3)`，两个数组扩展为 `(3, 3)`，因此 `result.shape == (3, 3)`。

它可能不是开发者想要的结果，因为开发者可能只是想给三行逐项相加，而不是生成一个 3×3 矩阵。

---

## D. NumPy 编程参考

### D1

```python
import numpy as np

X = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]
)

assert X.shape == (4, 3)
```

### D2

```python
x = np.array([2.0, -1.0, 3.0])
y = np.array([4.0, 5.0, -2.0])

assert np.array_equal(x + y, np.array([6.0, 4.0, 1.0]))
assert np.array_equal(x * y, np.array([8.0, -5.0, -6.0]))
assert np.isclose(x @ y, -3.0)
```

### D3

参考实现见：[`../../code/lesson001_vectors.py`](../../code/lesson001_vectors.py)。核心是先检查输入维度和特征数，再计算：

```python
predictions = X @ w + b
```

### D4

```python
one_dimensional = np.array([1.0, 2.0, 3.0])
row = np.array([[1.0, 2.0, 3.0]])
column = np.array([[1.0], [2.0], [3.0]])

assert one_dimensional.shape == (3,)
assert one_dimensional.T.shape == (3,)
assert row.T.shape == (3, 1)
assert column.T.shape == (1, 3)
```

一维数组没有第二个轴，因此转置不会交换任何轴。

### D5

```python
A * B  # 对应元素相乘
A @ B  # 行与列做点积
```

即使输入 shape 相同，两种运算的定义也完全不同。

### D6

```python
loop_result = np.array([row @ w + b for row in X])
vectorized_result = X @ w + b

np.testing.assert_allclose(loop_result, vectorized_result)
```

向量化写法通常更短，也能使用底层优化过的线性代数实现。

---

## E. 错误诊断

### E1

1. `X` 的列数是 20，但 `w` 的长度是 100，矩阵乘法中间维度不匹配；
2. `w` 应为 `(20,)`；
3. 错误把样本数 100 当成了特征数。

### E2

1. `w` 的 `(4,)` 可以广播到 `X` 的每一行，因此代码能运行；
2. 结果仍有 4 列，没有对每个样本的加权特征求和；
3. 正确写法：

```python
predictions = X @ w
```

结果 shape 为 `(3,)`。

### E3

1. `error.shape == (8, 8)`；
2. 每个预测值会与全部 8 个目标值相减，而不是一一对应，损失的语义错误；
3. 修复方式：

```python
target = target.reshape(-1, 1)
error = prediction - target
```

或者：

```python
prediction = prediction.reshape(-1)
error = prediction - target
```

应根据模型约定统一采用一维或二维列向量。

### E4

`x` 是一维数组，`x.T.shape` 仍为 `(3,)`。真正的行向量可以写成：

```python
row = x.reshape(1, -1)
```

或：

```python
row = x[np.newaxis, :]
```

### E5

示例断言：

```python
assert X.ndim == 2
assert w.ndim == 1
assert X.shape[1] == w.shape[0]
```

还可以检查输出：

```python
predictions = X @ w + b
assert predictions.shape == (X.shape[0],)
```

---

## F. 综合任务参考思路

可设计：

```python
X = np.array(
    [
        [5.0, 85.0, 80.0, 3.0],
        [3.0, 92.0, 70.0, 8.0],
        [8.0, 78.0, 90.0, 2.0],
    ]
)

w = np.array([2.0, 0.6, 0.3, -1.5])
b = 5.0
scores = X @ w + b
```

权重含义：

- 工作年限为正权重：经验越多，得分越高；
- 技术面试为正权重；
- 沟通得分为正权重；
- 距离为负权重：距离越远，得分越低。

注意：工作年限、面试分数和距离的数值尺度差异较大，权重不能只比较绝对大小。后续学习特征标准化和优化时，会进一步处理这个问题。