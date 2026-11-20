# CP2 · A3 眼底视野实验 v1

这是独立的冻结模型对照，不是 A3 的替换版。数据只取 APTOS 原验证集（527 张）；脚本不训练、不读取 APTOS test 或 IDRiD 图像，也不修改 Figma、Gradio、既有模型及评估。使用已有的 384×384 RGB PNG、原有 Python 环境，以及原模型已记录的权重。FOV mask 估计的是可见眼底的外轮廓，不是病灶标注。

## 在 Windows Git Bash 运行

下载 ZIP 后，在 CP2 根目录解压。ZIP 只包含新的 `experiments/aptos_fov_v1/` 目录。下面假设下载到了默认 Downloads 文件夹；`unzip -n` 会跳过任何同名旧文件。保留这两级目录，勿把脚本单独拷贝进 CP2 根目录。

```bash
cd /c/CP2
unzip -n ~/Downloads/CP2_A3_FOV_Pilot_v1.zip
python -B experiments/aptos_fov_v1/run_fov.py preview
```

1. **preview（建议先做）**：只用 RGB 检查全部 527 张验证图，不加载权重。查看 `sample_01.png` 的青绿色轮廓以及所有 `flagged_*.png`；对照原图，检查暗部、眼底边缘是否被切掉，外围相机黑框是否被误纳入。`mask_audit.csv` 有逐张结果；`masks/图像ID.png` 是每张的二值遮罩。没有警告不代表人工确认正确。空白或无法识别的照片会记为失败，不会偷偷回退到整张画布。
2. **smoke**：用预先按 ID 哈希抽取的 3 张验证图和正式 A3 权重，跑通三个分支，逐张核对原版预测，并产生 Top 10% 对照图。不发布准确率。原版应能复现训练阶段已保存的验证预测。
3. **compare**：用同一权重给全部 527 张验证图做配对比较。先检查 preview 和 smoke；发现切边问题请保留这次记录，勿把有问题的图移出队列再报分数。

```bash
python -B experiments/aptos_fov_v1/run_fov.py smoke
python -B experiments/aptos_fov_v1/run_fov.py compare
```

每次运行都在 `experiments/aptos_fov_v1/runs/<UTC时间>_<模式>_<随机ID>/` 新建目录。旧结果不会覆盖。没有覆盖、续写或选择任意输出路径的开关。若失败，保留当次错误和已有逐张记录；重跑另开目录。输出均已被本实验目录内的 .gitignore 排除。

沿用原 CP2 环境（Python、NumPy、Pillow、SciPy、PyTorch、torchvision、PyG 等）。原训练环境的版本和上游代码、数据清单必须匹配。运行前无需获取新权重；如果找不到 SciPy，先记录错误，不要为此改动正在使用的正式环境。可用 `--device cpu` 显式选择 CPU。显卡只用于模型；分区仍在 CPU 上执行。

## 三组具体比较什么

| 分支 | 区域定义 | 它能回答什么 |
|---|---|---|
| original_A3 | 完整复用原版 boundary-SLIC 和 A3 forward，包括黑色区域 | 在同一批验证图上复现基线 |
| original_regions_clipped | 从同一份原分区剔除视野外像素和完全在外的节点；若裁剪令一个区域断开，分开计数 | 保留原分区的起点，只改变区域对眼底的覆盖及相邻关系 |
| FOV_resegmented | 只在估计视野内放置 256 个初始种子；分配、更新、连通性合并均不使用视野外像素作成员 | 把分区预算也集中到眼底后，预测和区域排序怎样改变 |

两个 FOV 分支都保持相同 RGB 和 DenseNet 特征提取。区域池化、图邻接、GAT、ViT 和 Rollout 仅接收有效区域，无背景节点。面积特征仍按原 384² 归一化，质心坐标系不变。DenseNet 感受野仍可看到边缘，不能声称整个模型对背景绝对不敏感。重新分区用了不同的种子位置、有效面积对应的空间步长及视野内边界强度归一化，因此第三组的差异不能全部归因于删去黑边。256 是目标种子数，连通性处理后的实际区域数可不同。

## 看什么结果

- **preview**：`summary.json`、`mask_audit.csv`、代表性缩略图及所有标记图。
- **smoke / compare**：`paired_predictions.csv` 包含每张的真值、三个分支的预测和五类概率、实际区域数、Top 10% 所选像素和落在估计 FOV 外的像素数。图中青绿色表示所选视野内像素，粉色表示所选视野外像素，不代表糖网病灶。Top 10% 按实际区域数向上取整；像素覆盖率可不等于 10%。
- **完整 compare 的 summary.json**：三组 QWK、macro-F1、accuracy、混淆矩阵、相对原版差值及预测改变张数。同时保留每张 mask 的警告。只有 527 张都配对成功、所有原版预测按图像 ID 与 A3 epoch 21 的存档一致，才输出比较指标。脚本还检查了 A3 权重哈希和已读取的上游文件。标记为 `REVIEW_MASKS` 的结论仍须先看遮罩。
- 两个 FOV 分支的“所选视野外像素数 = 0”是设计保证，不是准确率改善，也不是病灶定位验证。验证集已参与原模型选择；本实验是开发对照，不是新的独立测试。当前官方 IDRiD 指标保持原值。

## 方法来源

- Papadopoulos, Topouzis & Delopoulos (2021), [An interpretable multiple-instance approach for the detection of referable diabetic retinopathy in fundus images](https://doi.org/10.1038/s41598-021-93632-8). 提供分类前过滤视野外图块的思路；本实验没有沿用其 50% 图块阈值。
- Irving (2016), [maskSLIC](https://arxiv.org/abs/1606.09518). 提供 mask 内生成 superpixel 的方法依据。这里的代码是为 CP2 写的变体，不是该论文的官方实现。它复用 CP2 的边界检测、栅格路径、连通分量和已训练的神经模块。
