Corner Proposal Network for anchor-free, two-stage object detection

# A. Abstract



# B. Introduction

## B.1 目标定义

尽可能多的找到目标物体 + 尽可能准确的分配标签 高recall + 高precision

## B.2 相关方法分析

|目标| 相关方法|初步结论|
|---|---|---|
|提高物体Recall| anchor-free vs anchor-based| anchor-based受限于anchor设定，对于常规物体有较好表现；但是，对于特殊size or shape的物体，召回能力不如anchor-free|
|提高物体precision| one-stage vs two-stage| two-stage方法引入类似多层筛选/回归方式，并且roi pooling可以更好滴完成feature和proposal的assign, 因此效果更佳|

## B.3 网络设计

|环节|参考方法|亮点 or 改进|
|---|---|---|
|Proposal/RPN| Anchor-free中cornerNet| 不进行keypoints的配对，而是选择罗列所有可能的配对|
|OBN|Faster RCNN|增加了基于size的过滤classifier, 此处不考虑类别，只是过滤不合规的size可能性|


# C. Related Work

## C.1 use anchor or not

### C.1.1 Anchor-based methods

#### 定义

placing a large number of anchors, which are regional proposals with different but fixed scales and shapes, and are uniformly distributed on the image plane

These anchors are considered as proposals and an individual classifier is trained to determine the objectness as well as the class of each proposal. 

#### 相关文献

| 提升方向|相关论文 |
|---|---|
|basic quality of regional features extracted from the proposal|more powerful network backbones[13,36,14]; using hierarchical features to represent a region[23,34,27]|
|arriving at a better alignment between the proposals and features| align anchors to features[46,47], align features to anchors[7,5], adjust the achors after classification[34,27,3]|

### C.1.2 anchor-free methods

#### 定义

do not assume objects to come from uniformly distributed anchors

#### 相关文献

|提升方向|论文|
|---|---|
|keypoint-based: 预测多组关键点 + 构建关键点间联系| CornerNet[19], CenterNet[8], ExtremeNet[49]|
|anchorpoint-based：预测多组关键点+vector表示[比如 width/height or distance]|FCOS[39], CenterNet[48], FoveaBox[17], SAPD[50]|

## C.2 use two-stage or one-stage

### C.2.1 two-stage detector

#### 定义

RPN + OBN; RPN部分能极大缓解正负样本不均衡问题 + feature alignment问题

#### 相关文献

34,12,3,31,22

acclerate: 使用RPN删除过滤大量FP； 6，23，12，3，31

### C.2.2 one-stage detector

33，27，24，46，5

# D. Methods

The goal is to locate a few bboxes and assign each one with a class label

## D.1 初始框架选择讨论

[1] use anchor-free or anchor-based methods for proposal extraction

[2] use two-stage or one-stage methods for determine the class of proposals

### D.1.1 Anchor-free or anchor-based methods

#### 方法说明

|方法|说明|
|---|---|
|anchor-based methods|each anchor is associated with a specific position on the image. 最终形状和anchor的关联性极大，虽然box regression会进行轻微调整，但是整体还是非常想过的|
|anchor-free methods|不受限于size, 直接预测size; determine its geometry and/or class afterward|

#### 观点说明&论证

core point: anchor-free methods have better flexibility of locating objects with arbitrary geometry, and thus a higher recal.

anchor-free方法对size和shape有更大灵活性，相对可以有更高recall

|优点|说明|
|---|---|
|减少anchor数量|铺设的anchor大多数都无真正对应的物体|
|提升效率|anchor设置时候只考虑常见shape和size,对于特殊size和shape考虑少|



### D.1.2 Two-stage or one-stage detector




## D.2 框架设计

### D.2.1 Step1: 使用Corner Keypoints完成proposals提取

### D.2.2 Step2: 利用two-step classification 进行proposals过滤

## D.3 inference设计

# E. Experiments

## Details

## ablation study

### compared with detectors

### classification improves precision

### inference speed


# F. 下一步阅读计划

## adjust anchors after classification

Cascade rcnn: Delving into high quality object detection.




