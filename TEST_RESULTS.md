# 测试证据

日期：2026-11-26  
环境：Python 3.12.14、NumPy 2.3.5、Pillow 12.3.0，纯 CPU。

## 1. 官方分阶段检查

命令：`python check.py --stage all`

结果：**8/8 PASS**

| 阶段 | 结果 | max_abs | 容差 |
| --- | --- | --- | --- |
| assets | PASS | — | — |
| preprocess | PASS | 0 | 2e-6 |
| patches | PASS | 0 | 2e-4 |
| tokens | PASS | 4.005e-05 | 2e-4 |
| block0 | PASS | 7.063e-06 | 2e-4 |
| last_tokens | PASS | 4.768e-05 | 2e-4 |
| output | PASS | 5.439e-07 | 2e-5 |
| batch | PASS | 4.619e-07 | 2e-5 |

## 2. 契约测试

命令：`python tests/test_encoder_contract.py`

结果：**7/7 OK**（约 12.3s）

覆盖：patch/head 顺序、B=1/2/5 输出契约、batch/排列/重复一致性、确定性、
CLI ID 顺序、无参考文件运行。

## 3. 变异验证

- 把 `patchify` 改成返回全零 → `test_patchify_patch_and_channel_order` FAIL
  （3071/3072 个值不一致），测试能抓住错误。
- 把 `split_heads` 改成错误 reshape → `test_split_heads_preserves_token_and_head_order`
  FAIL（`[2,3]` vs `[8,9]`），测试能抓住错误。
- 恢复实现后 7/7 通过。

## 4. CLI 验收

- `python run.py --output output_tokens.npz` → `(10,512)`，float32，全有限，
  单位长度最大误差 `1.19e-7`。
- `python run.py --images assets/images/image_02.jpg assets/images/image_01.jpg` →
  `(2,512)`，ID 顺序为 `image_02, image_01`。
- 从子目录运行单图 → `(1,512)`，路径解析正确。

## 复现说明

`weights/image_encoder.npz` 和 `assets/reference_outputs.npz` 不随仓库提交；
按 README 从课程附件放回原位后，以上命令可复现。
