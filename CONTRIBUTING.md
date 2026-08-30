# 贡献指南

欢迎补充课程、练习、代码、测试和错误诊断案例。本项目首先追求教学正确性和学习闭环，其次才是内容数量。

## 内容原则

新增内容应尽量满足：

1. 先说明要解决的问题，再给定义；
2. 公式中的变量必须先定义；
3. 关键表达式必须标注 Shape；
4. 至少包含一个可手算的小例子；
5. 说明它与机器学习模型的连接；
6. 代码应能直接运行，并包含输入检查；
7. 手写梯度应有数值梯度验证；
8. 至少记录两个常见误解或失败场景；
9. 不为了简洁而省略关键推导步骤；
10. 不把“调用现成 API”当作理解数学的替代品。

## 课程文件结构

```text
lessons/NNN-topic.md
exercises/NNN-topic.md
code/lesson_NNN_topic.py
tests/test_lesson_NNN_topic.py
```

课号使用三位数字，文件主题使用小写英文和下划线或连字符，并保持同一课的命名可对应。

## 课程正文模板

```markdown
# 第 N 课：主题

## 本课要解决的问题
## 学习目标
## 前置知识
## 直觉解释
## 严格定义
## 符号与 Shape
## 手算示例
## 关键推导
## NumPy 实现
## 与机器学习模型的连接
## 常见错误与诊断
## 验收问题
## 完成标准
```

章节可以根据主题调整，但不要省略问题、Shape、模型连接和诊断部分。

## 练习要求

每组练习尽量包含：

- 直接计算题；
- 公式推导题；
- 概念解释题；
- 编程实现题；
- 错误诊断题；
- 一个与真实数据或工程场景有关的开放题。

练习不应该只检查公式记忆。至少有一道题要求学习者解释为什么。

## Python 代码规范

- 支持 Python 3.11 及以上；
- 使用类型注解和简明 Docstring；
- 关键函数验证 Shape、空输入和非有限值；
- 默认使用 NumPy 展示数学实现；
- 示例应小而确定，避免依赖网络和大型数据集；
- 不静默修复含义不明确的输入；
- 浮点比较使用容差；
- 数值敏感操作使用稳定形式。

## 本地验证

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/lesson_001_vectors.py
pytest -q
```

提交前确认：

- [ ] 新增 Markdown 链接有效；
- [ ] 数学公式可以正常渲染；
- [ ] 示例中的计算结果正确；
- [ ] 测试覆盖正常输入和至少一个错误输入；
- [ ] 没有把本地数据、虚拟环境或生成文件提交到仓库；
- [ ] README 或课程目录已添加入口。

## Commit 建议

使用明确、单一目的的提交信息：

```text
lesson: add matrix multiplication lesson
exercise: add gradient practice set
feat: implement stable softmax
fix: correct logistic regression gradient
 test: cover zero-vector validation
docs: clarify MLE and cross-entropy relation
```

每次变更尽量只解决一个问题，便于 Review 数学正确性和教学意图。