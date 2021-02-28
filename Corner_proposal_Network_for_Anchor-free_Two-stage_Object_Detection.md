Corner Proposal Network for anchor-free, two-stage object detection

A. Abstract



# B. Introduction

## 分析 or 引言

### 目标定义

尽可能多的找到目标物体 + 尽可能准确的分配标签 高recall + 高precision

### 相关方法分析

|目标| 相关方法|初步结论|
|---|---|---|
|提高物体Recall| anchor-free vs anchor-based| anchor-based受限于anchor设定，对于常规物体有较好表现；但是，对于特殊size or shape的物体，召回能力不如anchor-free|
|提高物体precision| one-stage vs two-stage| two-stage方法引入类似多层筛选/回归方式，并且roi pooling可以更好滴完成feature和proposal的assign, 因此效果更佳|

### 网络设计

|环节|参考方法|亮点 or 改进|
|---|---|---|
|Proposal/RPN| Anchor-free中cornerNet| 不进行keypoints的配对，而是选择罗列所有可能的配对|
|OBN|Faster RCNN|增加了基于size的过滤classifier, 此处不考虑类别，只是过滤不合规的size可能性|


# C. Related Work

## Anchor-based methods

### 定义

placing a large number of anchors, which are regional proposals with different but fixed scales and shapes, and are uniformly distributed on the image plane

These anchors are considered as proposals and an individual classifier is trained to determine the objectness as well as the class of each proposal. 

### 相关文献

| 提升方向|相关论文 |
|---|---|
|basic quality of regional features extracted from the proposal|more powerful network backbones[13,36,14]; using hierarchical features to represent a region[23,34,27]|
|arriving at a better alignment between the proposals and features| align anchors to features[46,47], align features to anchors[7,5], adjust the achors after classification[34,27,3]|

## anchor-free methods

### 定义

do not assume objects to come from uniformly distributed anchors

### 相关文献

|提升方向|论文|
|---|---|
|keypoint-based: 预测多组关键点 + 构建关键点间联系| CornerNet[19], CenterNet[8], ExtremeNet[49]|
|anchorpoint-based：预测多组关键点+vector表示[比如 width/height or distance]|FCOS[39], CenterNet[48], FoveaBox[17], SAPD[50]|

D. Methods


E. Ablation Study

F. Discussion

# H. 下一步阅读计划

## adjust anchors after classification

Cascade rcnn: Delving into high quality object detection.




