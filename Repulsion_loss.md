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

### Object 
