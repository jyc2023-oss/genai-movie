# GenAI Feature-Fusion Movie Rating Pipeline

本项目严格按照《大模型与GenAI课程设计策划书》实现，围绕“GenAI即特征工程师”的思路搭建了零成本（Google Colab + Hugging Face开源模型）流水线。核心目标：利用LLaVA-1.5-7B从电影海报抽取美学关键词、用Qwen1.5-7B-Chat为剧情摘要打感知分，再与传统结构化特征融合，最后用XGBoost完成电影评分预测，并通过A/B实验验证增益。

## 目录结构
```
.
├── requirements.txt            # Colab/本地安装依赖
├── src/
│   ├── config.py               # 全局路径与模型配置
│   ├── data_prep.py            # Phase 2：结构化特征工程
│   ├── download_posters.py     # Phase 0：海报批量下载
│   ├── feature_fusion.py       # Phase 3：特征拼接
│   ├── train.py                # Phase 4：A/B训练与可视化
│   └── genai/
│       ├── poster_llava.py     # Phase 1A：LLaVA提取美学关键词
│       └── plot_qwen.py        # Phase 1B：Qwen量化情节评分
├── artifacts/                  # 运行后生成的中间数据、图表
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
└── docs/
    └── colab_run.md            # （下节步骤更详细版本）
```

## 一键环境（Google Colab）
1. 打开 [colab.research.google.com](https://colab.research.google.com/) → `Runtime > Change runtime type > GPU (T4)`。
2. 在首个代码单元执行：
   ```bash
   !git clone <your_repo_url> genai-movie
   %cd genai-movie
   !pip install -r requirements.txt
   ```
3. **Hugging Face鉴权**（访问LLaVA/Qwen模型必需）：
   ```bash
   from huggingface_hub import login
   login()  # 输入你的HF Token
   ```
4. （可选）挂载Google Drive持久化海报：
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## 分阶段运行指令
| Phase | 目的 | Colab 命令 |
|-------|------|-----------|
|0|下载海报，构建图像语料|`python src/download_posters.py --image-size w500`|
|2|生成结构化特征|`python src/data_prep.py`|
|1A|调用LLaVA抽取“美学关键词”|`python src/genai/poster_llava.py`|
|1B|调用Qwen生成“plot_novelty / emotion_complexity”|`python src/genai/plot_qwen.py`|
|3|合并传统 & GenAI 特征|`python src/feature_fusion.py`|
|4|XGBoost A/B 实验 & 可视化|`python src/train.py --test-size 0.2`|

> 建议顺序：Phase0 → Phase2 → Phase1A → Phase1B → Phase3 → Phase4。每一步都会把结果落在 `artifacts/` 下，可随时断点续跑。

### 重要提示
- Phase1A/1B 运行时间取决于Colab GPU排队情况，大约 2~3 小时即可完成 4k+ 样本；若需分批，可临时修改脚本中 `iterrows()` 的切片范围。
- 若某些海报缺失，脚本会自动跳过；最终 `poster_features.parquet` 仍会与结构化特征左连接。
- Qwen输出若非严格JSON，`plot_qwen.py` 会把原始回复写入 `raw_response` 方便排查。

## 产出物（位于 `artifacts/`）
- `datasets/*.parquet`：结构化、LLaVA、Qwen以及融合后的超级特征表。
- `reports/metrics.json`：Baseline 与 GenAI 模型的 RMSE、增益 Δ。
- `reports/figures/rmse_comparison.png`：A/B 柱状图（Phase7 定量图）。
- `reports/figures/feature_importance.png`：GenAI 模型特征重要度（可直接插入报告）。

## 报告撰写建议
1. **方法论**：引用策划书“GenAI-as-a-Feature-Engineer”概念，简述三种特征来源（结构化/海报/剧情）。
2. **实验设置**：说明 Colab T4、4-bit 量化加载、XGBoost 参数、train/test 切分方式。
3. **结果展示**：贴 `rmse_comparison.png`、`feature_importance.png`，并在文字中突出 `RMSE_B < RMSE_A`。
4. **案例截图**：
   - LLaVA 关键词示例（《黑暗骑士》→ `dark, chaos, gothic...`）。
   - Qwen JSON 输出示例（《盗梦空间》→ plot_novelty / emotion_complexity）。
5. **洞察**：结合特征重要度说明 `plot_novelty`、`poster_keywords` 的贡献，呼应“量化艺术”主题。

若需更细的Colab逐单元脚本，请阅读 `docs/colab_run.md`（已把每一步拆成可直接复制的代码块）。
