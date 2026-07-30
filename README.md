# 量化交易：AI辅助的金融交易策略

研究生课程完整项目，包含从数据获取到AI交易决策的全流程量化分析工具。分析标的：**比亚迪 002594.SZ**。

## 在线预览

👉 **[交互式指标分析工具 (GitHub Pages)](https://david-casia.github.io/AI-quant/)**

> 支持技术指标(RSI/MACD/BOLL/KDJ)、双均线/海龟/融合策略、ML分类预测、MLS选股回测、AI综合交易建议——全部在浏览器中实时运行。

## 项目结构

```
├── indicator-tool/          # 🌐 交互式网页工具 (核心交付物)
���   └── index.html            #    单文件，双击即用，无需安装
│
├── TASK1/                    # 数据获取 + 收盘价可视化
├── TASK2/                    # 技术指标 (RSI/MACD/BOLL/KDJ)
├── TASK3/                    # 双均线策略 + 参数对比
├── TASK4/                    # 海龟策略 (唐奇安通道 + ATR止损)
├── TASK5/                    # ML分类模型 (LR/决策树/随机森林 + ROC/AUC)
├── TASK6/                    # ML选股策略 (48只A股 + 12因子 + 季度回测)
├── TASK7/                    # JoinQuant平台策略部署 + 20组参数优化
├── TASK8/                    # 综合复盘报告 (18页学术报告)
│
└── .workbuddy/memory/        # 项目长期记忆文件
```

每个 TASK 文件夹包含：
- Python 脚本 (数据获取/分析/绘图)
- 生成的图表 (PNG)
- 编译报告 (HTML + PDF)
- 数据文件 (CSV)

## 快速上手

### 方式一：直接使用网页工具（推荐）

1. 打开 `indicator-tool/index.html`（双击即可，浏览器中运行）
2. 页面自动加载比亚迪前复权日线数据
3. 调整各指标参数，观察实时图表变化
4. 训练 ML 模型预测次日涨跌
5. 运行 MLS 选股策略回测
6. 参考 AI 综合交易建议

### 方式二：运行 Python 脚本

```bash
# 安装依赖
pip install pandas numpy matplotlib tushare weasyprint

# 进入某个 TASK 目录，运行数据处理脚本
cd TASK5
python plot_ml.py      # 生成ML图表
python build_pdf.py    # 生成PDF报告
```

> **注意**：每个 TASK 中的 `build_pdf.py` 会生成 HTML 和 PDF 双版本报告。如果 Windows 上缺少 GTK，PDF会自动通过 Edge headless 生成。

## 技术栈

- **数据**: Tushare (前复权日线)
- **分析**: pandas, numpy
- **可视化**: matplotlib, Chart.js
- **策略**: 双均线(金叉/死叉), 海龟(唐奇安通道+ATR)
- **ML**: 逻辑回归, 决策树, 随机森林 (纯JS实现 + Python sklearn对照)
- **报告**: weasyprint / Edge headless → PDF
- **部署**: JoinQuant 策略平台, GitHub Pages

## 关键数据约定

| 项 | 值 |
|------|------|
| 复权方式 | 前复权 (qfq) |
| 分析标的 | 比亚迪 002594.SZ |
| ML 特征 | 17 维度 (价格/波动/量价/动量) |
| MLS 因子 | 12 维度 (收益率/偏离/波动/量比/动量/振幅) |
| 中国股市约定 | 涨红跌绿 |

## Disclaimer

本项目仅供学术研究和学习参考，不构成投资建议。
