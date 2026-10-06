# Groundwater Forecasting Baselines | 地下水位预测通用基线

[![Tests](https://github.com/fangxiaoyu907-del/groundwater-forecasting-baselines/actions/workflows/tests.yml/badge.svg)](https://github.com/fangxiaoyu907-del/groundwater-forecasting-baselines/actions/workflows/tests.yml)

从毕业论文实验工作区中整理的、**不含研究数据的通用时序回归代码**，用于展示模型结构、训练流程和可复现工程实践。

这里只发布可独立分离的标准基线和通用数值工具。它不是三篇论文的完整代码集合，**不是找回的第一、第二篇原始源码，也不声称复现论文结果**。原工作区中的模型文件本身已标明为重构实现；公开版进一步做了删减、独立化与运行兼容性整理。来源和改动见 [docs/PROVENANCE.md](docs/PROVENANCE.md)。

## 包含什么

- 标准 GRU、BiGRU、BiGRU + 单头时间注意力，支持直接多输出回归。
- 持续性预测、训练集标准化的 Ridge 增量预测、MAE / RMSE / R² / 相对持续性技巧分数。
- 通用 GWO 数值搜索接口，保留原实验通用数学部分的历史候选精英存档变体；不是第一篇原始优化器。
- 因果历史窗口、按目标时间划分训练/验证/测试、跨边界多步标签保护、训练集专用标准化。
- 只在内存生成的合成信号演示，以及结构、梯度、模型存取和输入校验测试。

不包含未公开的研究方法、原始数据、真实井点标识、论文正文、实验结果、训练权重、拟合统计量、私有配置或旧文件历史。

## 运行

使用 Python 3.12，在本仓库目录执行。先创建独立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_demo.py
```

上述 NumPy-only 安装会明确跳过神经网络测试。完整验证与神经网络演示：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-neural.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_demo.py --neural --epochs 3
```

Linux / macOS 将解释器路径换为 `.venv/bin/python`。公开版固定 NumPy 2.2.6、TensorFlow 2.20.0，以 CPU 小规模演示为目标；原研究环境及结果不属于本仓库的验证范围。GPU 设置另见 [TensorFlow 官方安装说明](https://www.tensorflow.org/install/pip)。

演示默认仅打印合成信号指标，**不写出数据、权重或参数文件**。GWO 演示优化数学函数，不执行论文的私有模型调参。

## 模型调用

```python
from models import build_model

model = build_model(
    "attention", lookback=12, n_features=2, n_outputs=3,
    units=[16], dropout=0.2, seed=42,
)
# model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
#           shuffle=False, epochs=20)
```

`X` 形状为 `(样本数, 历史窗口长度, 特征数)`；`y` 为 `(样本数, 输出数)`。输出数不是预测提前量，需由调用者定义标签与提前量顺序。模型直接输出目标值；Ridge 基线则预测相对起报时刻观测的增量，两者不能混淆。

时间注意力对**时间轴**做 softmax，然后逐时加权并展平；不是加权求和输出头。BiGRU 的双向操作只在起报时刻之前的历史窗口内部进行，不需访问未来观测。

## 数据安全与评估边界

仓库没有真实或脱敏后的研究数据。合成信号也不来自真实观测的拟合或扰动。

- 数据必须由使用者合法持有并留在本地；仓库不提供下载接口和私有数据路径。
- 必须先按时间划分，再仅用训练段拟合预处理；不能随机拆分重叠窗口，或用未来观测插值历史输入。
- 多步样本仅在全部标签属于同一划分时保留；验证/测试窗口可以使用起报时刻前已知的历史。
- 合成指标仅检查流程能否运行，不代表实际水位预测精度、泛化能力或论文结论。公开演示并未完成真实场景调参、部署或业务验证。
- 自己运行训练后，权重、归一化器、日志、预测和指标也可能泄密；不得直接提交。

`.gitignore` 只是辅助保护，不替代逐文件审核。公开提交仅允许代码、依赖声明、文档和测试流程。

## 目录

```text
models.py                三种通用神经网络基线
baselines.py             持续性、Ridge 增量预测、评价指标
gwo.py                   通用目标函数搜索
windows.py               历史窗口、时序划分、训练专用标准化
run_demo.py              内存合成信号演示
tests/                   自动测试
docs/PROVENANCE.md       来源、删减与重构边界
docs/VALIDATION.md       验证记录
```

## 许可与履历

按作者当前选择，暂不添加开放源代码许可证；本仓库是公开代码展示，不等同于已授予开放源代码使用许可。复用或再分发请先联系作者。开发、整理和发布日期以实际记录为准，不回填历史开发经历；简历应表述为“通用预测基线整理与工程化展示”，不可表述为已公开三篇论文全部原始模型或复现全部论文指标。
