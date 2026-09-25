# TPS Decoder Evaluation

这是从论文最终实验脚本整理出的、可迁移的开源版本。程序评估一张带水印图像在以下扰动下的解码鲁棒性：

- 随机 TPS 几何扰动
- JPEG 压缩
- CompressAI VAE 重建
- 可选的 GradNorm/UnMarker 梯度攻击
- 可选的多步 Pareto beam search

仓库只包含代码和配置示例，不包含论文实验输出、缓存、模型权重或任何机器专属绝对路径。

## 目录结构

```text
src/
  eval_tps_decoder_curve.py       # 实验编排、攻击曲线、决策和批量模式
  attack_topk_decoders.py         # 解码器加载和评估指标
  attack_topk_decoders_gradnorm.py# low-frequency UnMarker/质量攻击后端
  decoder_entropy_ranking.py      # 候选解码器的信息熵排序
  crop_resize_image.py            # 可选的输入预处理
scripts/
  run_single.sh                   # 单图入口
  run_batch.sh                    # 数据集/多 GPU 入口
config/
  default.env.example             # 路径和最小配置示例
tests/
  test_smoke.py                   # 不加载模型的导入和参数检查
```

## 安装

建议使用 Python 3.10 或更高版本，并根据本机 CUDA 版本安装对应的 PyTorch。然后安装本项目依赖：

```bash
python -m pip install -e .
```

候选解码器的模型实现不是本项目重新实现的内容，需要放在外部目录，并通过 `config/default.env.example` 中的环境变量指向它们。至少需要：

- `WATERMARK_MODELS_ROOT`：MBRS、HiDDeN、CIN、FIN、PIMoG、TrustMark、InvisMark、RoSteALS、LightweightMark、VideoSeal 等模型仓库的父目录
- `ENCODER_CLASSIFI_ROOT`：候选排序所需的 `decoder_entropy_ranking.py` 依赖和模型数据
- 各模型对应的 checkpoint 和配置文件

## 单图运行

```bash
source config/default.env.example   # 也可以复制后按本机路径修改
export WATERMARKED_IMAGE=/path/to/encoded.png
export TRUE_DECODER=cin
export OUTPUT_DIR=outputs/example
bash scripts/run_single.sh
```

脚本默认使用 entropy top-k 选择候选解码器，并运行 TPS、JPEG、VAE 和顺序搜索。只想做非对抗扰动时：

```bash
ENABLE_ADVERSARIAL_ATTACK=false bash scripts/run_single.sh
```

## 批量和多 GPU 运行

数据目录需要按水印方法分组，每个样本目录中包含 `encoded.png`，例如：

```text
data/
  CIN/000000000001/encoded.png
  FIN/000000000002/encoded.png
```

运行批量评估：

```bash
export DATA_ROOT=/path/to/encoder_residual_stats_500
export OUTPUT_ROOT=outputs/batch
export IMAGES_PER_CATEGORY=10
bash scripts/run_batch.sh
```

多 GPU 使用逗号分隔的物理 GPU 编号：

```bash
GPU_IDS=0,1,3 bash scripts/run_batch.sh
```

## 清理说明

原始实验脚本混合了历史实验、输出目录和机器专属路径。本版本已移除这些内容，并把路径全部改为环境变量或项目相对路径。

公开版还彻底移除了 decoder bit loss、频域/reference-residual loss、DnCNN 去噪图以及对应的参数和输出。对抗阶段只保留论文最终使用的 low-frequency UnMarker 梯度、图像质量约束和 LPIPS 选择约束；BER/bit accuracy 仍作为攻击后的评估指标计算，但不会参与梯度更新。

## 输出

单图输出目录通常包含 `metrics/`、`decision/`、`verification/` 和 `plots/`。批量模式额外生成 `manifest.tsv`、`per_image_results.csv`、`category_success_rates.csv` 和 `batch_summary.json`。这些结果文件已被 `.gitignore` 排除，避免误提交大体积实验产物。

## 测试

```bash
pytest -q tests/test_smoke.py
```

该测试只检查代码导入和参数默认值，不需要加载任何水印模型权重。
