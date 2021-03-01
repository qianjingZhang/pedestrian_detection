# Repulsion Loss: Detecting Pedestrains in a Crowd

## Abstract

- 针对问题： 密集行人检测
- 方法动机：
  - 针对target box: attract. 靠近
  - 针对 other bbox: repulsion, 惩罚  
- 使用方法： 提出针对bbox regression的repulsion loss


## Introduction

### 问题说明&定义

- 遮挡： inter-class & intra-class
  - inter-class: 物体被其他杂物或者其他类东西遮挡
  - intra-class: 物体被同一类物体遮挡， crowd occlusion

- 主要影响：难点在于 pedestrian localization
  - T&B公用相同的特征，但是却需要回归2个不同的框，此处产生fused
  - crowd occlusion 对NMS阈值更加敏感 

### 策略

- 考虑target的同时考虑surrounding bboxes
- 如果bbox和non-target的object产生了interaction, 会被loss惩罚


### 贡献

- 实验性分析：We first experimentally study the impact of crowd occlusion on pedestrian detection. Specifically, on the CityPersons benchmark [33] we analyze both false positives and missed detections caused by crowd occlusion quantitatively, which provides important insights into the crowd occlusion problem
- 提出特殊loss: Two types of repulsion losses are proposed to address the crowd occlusion problem, namely **RepGT Loss and RepBox Loss**
  - RepGT Loss: 惩罚 预测框和其他gt产生了交集
  - RepBox Loss: 惩罚 预测框和其他不享有相同gt的anchor产生交集


## Related Work

### Object Localization


|论文|关键方法key| 说明|
| --- | --- | --- |
|10   |L1 Loss | Regression model + loss设置为 Euclidean distance|
|9    | Smooth L1 Loss| 考虑到差异远情况下的梯度收敛问题|
|24   | RPN| twice转换，收敛|
|15/29| IoU Loss+anchor-free| 收敛不单纯考虑diff,而是考虑iou，更加适应评测指标|
|4    | attraction and repulsion between object to capture the spatial arrangements of various object classes|


### Pedestrian Detection

|论文|方法key|说明|
| ---|---|---|
|[5,22,32]|传统方法，使用Integral Channel Features|sliding window + handcrafted features|
|[28,30]|deep features + ML methods|特征提取方面的优化|
|[23,27,34]|part-based methods|针对遮挡的专门优化|
|[13]|robustness of NMS|针对NMS的优化，使用了额外网络|


## Impact of crowd occlusion

### preliminaries

///

### Analysis on Failure Cases


#### Miss
|研究问题|评测说明|方案说明|
| --- | --- | --- |
|**occlusion定义**|occ = 1 - area(box_visible)/area(bbox)|occ>=0.1: occlusion case; occ >= 0.1 & IoU >= 0.1: crowd occlusion|

- **评测数据集**: 设置问题定义之后，划分成不同类型数据集； the reasonable-occ subset, consisting of 810 occlusion cases (51.3%) and the reasonable-crowd
subset, consisting of 479 crowd occlusion cases (30.3%).

- **不同评测指标**： 在不同recall/fp下， 由研究问题导致的detection占比


#### False Positive

FP分析采用统计分类的方式：Background(true fp), localization(reg fp) and crowd error(multi_fp)

看导致FP的crowd数量






