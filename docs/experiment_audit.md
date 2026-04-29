# BEV-TextCLIP 项目体检报告

## 当前方法概述

BEV-TextCLIP 目前实现为一个多模态 BEV 语义分割原型系统。

核心流程如下：

1. 多视角图像由 ResNet50 或 ViT 骨干网络编码。
2. 图像特征通过简化的 LSS 风格变换提升到 BEV 空间。
3. 点云由简化的 PointPillar 或 VoxelNet 风格编码器编码。
4. 图像 BEV 特征和点云 BEV 特征通过门控注意力融合模块进行融合。
5. 类别名称由 CLIP 文本编码器编码为类别嵌入。
6. BEV 特征通过 BEV-文本交叉注意力与文本嵌入进行交互。
7. 卷积分割头预测 BEV 语义标签。
8. 训练使用 focal loss、dice loss，以及简化的全局多模态对比损失。

目前最稳妥、最适合论文表述的主线是：

> 一个文本引导的多模态 BEV 语义分割框架，通过门控融合和 BEV-文本交互，将类别级语言先验注入融合后的相机-LiDAR BEV 特征中。

## 当前实验状态

根据已有训练日志 `train_wd002_ep45.log`，目前可见的最新一次训练完成了 45 个 epoch。

最终验证指标如下：

- Val Loss: 3.9930
- Val Accuracy: 0.7321
- Val mIoU: 0.2618

第 44 个 epoch 的代表性逐类 IoU 如下：

- car: 0.5346
- pedestrian: 0.3060
- driveable_surface: 0.7799
- sidewalk: 0.3534
- terrain: 0.4087
- manmade: 0.3934
- vegetation: 0.6654
- other: 0.3952
- truck: 0.0687
- bus: 0.0211
- construction_vehicle, bicycle, motorcycle, traffic_cone, barrier: 0.0000
- trailer: 验证集中为 N/A

这说明模型已经能够学习较大的静态区域和高频类别，但对长尾动态目标类别的建模仍然较弱。

## 主要优势

- 代码已经具备清晰的模块化结构，包括图像编码器、点云编码器、融合模块、BEV-文本注意力、损失函数、数据加载器、评估器和可视化模块。
- `train.py` 已经计算 mIoU 和逐类 IoU，这是语义分割实验中非常关键的指标。
- nuScenes lidarseg 标签已经被映射到 16 类训练标签体系，并栅格化为 BEV 标签。
- 模型支持通过本地 CLIP 或 HuggingFace CLIP 配置文本编码器。
- 现有曲线图和日志已经可以支撑初步的训练动态分析。

## 主要风险

### 1. 数据与评估流程不一致

`train.py` 使用的是 `NuScenesDataset` 和 `DataCollator`，但 `evaluate_open_world.py` 和 `evaluate_closed_set.py` 又分别定义了自定义的 nuScenes 数据加载器。

这些评估脚本在数据缺失或数据数量不匹配时可能生成随机标签或随机输入。它们适合做 demo，但不适合作为论文实验结果来源。

建议处理方式：

- 使用一套统一的标准评估流程，基于 `NuScenesDataset`、`DataCollator` 和 `BEVSegmentationEvaluator`。
- 对所有论文级评估脚本禁用随机 fallback 行为。
- 将评估指标保存为 JSON 或 CSV，便于直接生成实验表格。

### 2. 相机几何仍然是简化实现

当前 nuScenes 数据加载器中的 `_load_camera_params` 即使在真实样本文件存在时，也返回单位内参和单位外参。

这会削弱关于真实多视角几何建模或精确 LSS 投影的论证。

建议处理方式：

- 从 nuScenes 元数据中加载 `calibrated_sensor` 和 `ego_pose` 记录。
- 计算真实相机内参，以及 lidar 到相机或世界坐标系的外参。
- 在论文中明确说明图像分支使用的是真实标定，还是近似 BEV 引导。

### 3. 点云编码器仍处于原型级别

PointPillar 和 VoxelNet 的实现中包含 CPU/Numpy 转换和 Python 循环。对于研究原型这是可以接受的，但它并不等同于优化后的标准 PointPillars 实现。

建议处理方式：

- 如果不替换为标准实现，论文中建议描述为轻量级 PointPillar-style 编码器。
- 如果希望实验更强，可以接入成熟的 BEV/点云骨干网络，或者至少在当前代码中加入 point-only 版本进行对比。

### 4. 文本编码器在前向过程中是静态的

`CategoryEmbedder` 在初始化时将类别嵌入注册为 buffer。即使 `text_freeze=False`，当前前向传播也使用缓存的类别嵌入，并在 `BEVTextCLIP.forward` 中对其执行 `detach`。

这意味着当前实际训练路径并不会通过分割损失真正微调文本语义。

建议处理方式：

- 如果采用固定 CLIP 设置，应明确描述文本嵌入是冻结的语义原型。
- 如果希望采用可训练文本设置，需要在训练过程中重新生成嵌入，或者实现 prompt learning。

### 5. 对比学习弱于当前设计描述

`MultiModalContrastiveLoss` 中包含全局、局部和跨模态损失，但 `BEVTextCLIPLoss.forward` 目前为了节省显存，只实际使用了全局对比。

建议处理方式：

- 要么将当前损失明确表述为全局多模态对齐损失。
- 要么实现高效的采样式局部对比和类别感知的图像/点云/文本对比，再声称使用多层级对比学习。

### 6. 长尾类别尚未解决

当前最终结果中，多个小目标或稀有类别的 IoU 仍然为 0。

建议处理方式：

- 增加类别频率分析。
- 增加类别均衡损失权重或重加权策略。
- 同时报告全类别 mIoU 和高频类别 mIoU。
- 使用定性结果展示小类别失败案例。

## 推荐实验

### 主结果表

建议在同一数据加载器和同一评估器下比较以下方法：

1. 仅点云 BEV 分割
2. 仅图像 BEV 分割
3. 图像 + 点云拼接融合
4. 图像 + 点云门控融合
5. 图像 + 点云门控融合 + 文本注意力
6. 完整 BEV-TextCLIP，即文本注意力 + 对比损失

指标：

- mIoU
- 像素准确率
- 逐类 IoU
- 如果可行，报告 FPS 或推理时间

### 消融实验表

推荐消融项：

- 去掉文本交互
- 使用文本交互 + 冻结 CLIP 语义原型
- 使用随机 CLIP 嵌入
- 使用真实 CLIP 嵌入
- 去掉对比损失
- 使用对比损失
- 融合方式：addition、concatenation、gated attention
- Dropout 0.1 与 0.3 对比
- 冻结图像编码器与不冻结图像编码器对比

### 开放词汇实验

一个可信的开放词汇实验需要更谨慎的设置。

推荐协议：

- 只在部分类别上训练。
- 将若干类别从监督训练中保留出来，作为未见类别。
- 测试时，将保留类别的类别名称加入文本原型集合。
- 报告 seen-class mIoU、unseen-class mIoU 和 harmonic mean。

在移除随机 fallback 之前，不建议将当前 `evaluate_open_world.py` 的结果作为论文证据。

### 定性结果

有价值的可视化图包括：

- BEV 预测结果与 BEV 真值对比。
- 相机图像上下文 + BEV 结果。
- 针对特定类别 prompt 的 BEV 注意力可视化。
- 小类别失败案例，例如 bicycle、traffic cone、barrier。

## 论文写作方向

### 候选标题

BEV-TextCLIP: Text-Guided Multi-Modal BEV Semantic Segmentation for Autonomous Driving

### 核心贡献

1. 提出一个融合相机、LiDAR 和类别级语言先验的多模态 BEV 语义分割框架。
2. 设计一个门控 BEV 融合模块，自适应平衡图像 BEV 特征和点云 BEV 特征。
3. 设计一个 BEV-文本交互模块，将 CLIP 类别语义注入空间 BEV 特征。
4. 构建一套面向 nuScenes BEV 语义分割的实用训练与评估流程，并进行逐类 IoU 分析。

### 方法章节结构

1. 问题定义
2. 相机到 BEV 的图像编码器
3. 点云 BEV 编码器
4. 门控多模态 BEV 融合
5. 文本引导的 BEV 交互
6. 训练目标
7. 实现细节

### 实验章节结构

1. 数据集与评价指标
2. 实现细节
3. 主结果对比
4. 消融实验
5. 定性分析
6. 局限性

## 下一步建议

1. 将评估流程统一为一个论文级脚本。
2. 从评估中移除随机 fallback 行为。
3. 为模型或配置添加 baseline 开关。
4. 使用相同随机种子和相同数据划分运行受控消融实验。
5. 从 JSON/CSV 日志生成结果表格。
6. 基于当前架构起草方法章节。

