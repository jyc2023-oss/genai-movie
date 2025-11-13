# Google Colab 运行手册

> 目标：严格落地策划书的 Phase0-4。复制以下单元即可在 Colab 免费 GPU(T4) 环境中复现全流程。

## 0. 环境初始化
```python
!nvidia-smi
!pip install -q -U pip
!pip install -q -r requirements.txt
```
```python
from huggingface_hub import login
login()  # 输入你的HF Token，保证能访问 llava-hf/Qwen 模型
```
（可选）挂载 Google Drive 持久化海报：
```python
from google.colab import drive
drive.mount('/content/drive')
```

## 1. 数据与海报准备（Phase 0 + Phase 2）
```python
# Phase 0 - 下载海报
!python src/download_posters.py --image-size w500
```
```python
# Phase 2 - 结构化特征工程
default_args = ''
!python src/data_prep.py {default_args}
```

## 2. GenAI 特征工程（Phase 1）
```python
# Phase 1A - 海报关键词（LLaVA）
!python src/genai/poster_llava.py
```
```python
# Phase 1B - 情节评分（Qwen）
!python src/genai/plot_qwen.py
```
> 如果Colab会话有超时风险，可自行在脚本中添加 `--start-id / --end-id` 等参数实现分批跑（框架已经写成独立脚本，易于扩展）。

## 3. 超级特征拼接（Phase 3）
```python
!python src/feature_fusion.py
```

## 4. XGBoost 训练 & A/B 评估（Phase 4）
```python
!python src/train.py --test-size 0.2
```
运行结束后，可在 `artifacts/` 中下载：
- `reports/metrics.json`
- `reports/figures/rmse_comparison.png`
- `reports/figures/feature_importance.png`

## 5. 报告建议
1. 插入 RMSE 柱状图 & 特征重要度图，突出 GenAI 增益。
2. 截图 1-2 个 LLaVA / Qwen 输出作为“感知特征”示例。
3. 结合 `metrics.json` 撰写 A/B 对比文字：“Baseline RMSE = X，GenAI RMSE = Y，误差下降 Z%”。

至此即可完成策划书中的全部步骤。祝顺利！
