# Project 1：用 NumPy 复现图像编码器

## 项目说明

本项目只用 NumPy 实现 OpenCLIP ViT-B-32 的图像编码器前向计算：输入
`[B,3,224,224]` 的预处理图像，输出每张图一个单位长度的 `[B,512]` token。
模型参数已给出，不训练、不需要 GPU，也不需要安装 PyTorch/OpenCLIP。

## 目录结构

```text
image_encoder.py     实现（TODO-01～06，已完成）
helpers.py           LayerNorm、GELU、softmax、unit_vector 等课程提供函数
runtime.py / run.py  权重/图片加载与命令行入口
check.py             分阶段数值检查（对照独立 FP32 参考）
preprocess_image.py  resize / center crop / 归一化
config.json          模型结构
assets/              10 张示例图、manifest 与参考输出（参考输出不提交）
weights/             图像塔参数与 manifest（参数不提交）
tests/               7 项契约测试与测试证据
```

## 环境与权重

```bash
python -m pip install -r requirements.txt
```

仓库不提交 `weights/image_encoder.npz`（326 MB，超过 GitHub 单文件上限）和
`assets/reference_outputs.npz`（参考数据）。克隆后请从课程附件把这两个文件放回
`weights/` 和 `assets/`，`check.py` 才能运行。

## 运行与验证

```bash
python check.py --stage all
python run.py --output output_tokens.npz
python run.py --images assets/images/image_01.jpg assets/images/image_02.jpg --output my_tokens.npz
```

`check.py --stage all` 覆盖 assets、preprocess、patches、tokens、block0、
last_tokens、output、batch 八个阶段；输出容差为 output/batch `atol=2e-5,
rtol=2e-4`，内部 token `atol=2e-4, rtol=2e-4`。

## 实现要点

- `patchify`：`[B,3,224,224]` → `[B,3,7,32,7,32]` → transpose → `[B,49,3072]`，
  patch 顺序从左到右、从上到下。
- `embed`：patch 过无 bias 的 `[768,3072]` 投影，拼 CLS，再加 `[50,768]` 位置编码。
- `TransformerBlock`：pre-LN → 12 头注意力 → 第一次残差 → pre-LN → GELU MLP →
  第二次残差；共 12 层。
- 输出：取 CLS → post-LN → `visual.proj [768,512]`（不转置）→ `unit_vector`。

## 实现结果

| 阶段 | 结果 | max_abs |
| --- | --- | --- |
| assets | PASS | — |
| preprocess | PASS | 0.000e+00 |
| patches | PASS | 0.000e+00 |
| tokens | PASS | 4.005e-05 |
| block0 | PASS | 7.063e-06 |
| last_tokens | PASS | 4.768e-05 |
| output | PASS | 5.439e-07 |
| batch | PASS | 4.619e-07 |

## 额外测试

```bash
python tests/test_encoder_contract.py
```

7 项契约测试：patch/head 顺序、batch/顺序/重复一致性、确定性、CLI ID 顺序、无参考文件运行。
测试证据见 [TEST_RESULTS.md](TEST_RESULTS.md)。

## 仓库范围

仓库只保留源码、依赖、必要测试数据与测试；实验报告单独提交，不入库。
