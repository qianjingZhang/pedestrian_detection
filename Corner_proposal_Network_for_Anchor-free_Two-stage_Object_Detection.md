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

[1] follow CornerNet to locate an object with a pair of keypoints located in its top-left and bottom-reight corners

#### Step1.1: 计算keypoints

[2] For each class, 计算 two heatmaps [top-left heatmap and bottom-right heatmap], heatmap上每个点的值代表的是该点是此类型物体的概率

[3] 根据heatmap, 有两类型loss被计算。一、 Loss_det_corner 表示heatmap的关键点位置； Loss_offset 学习的是offset to the accurate corner position

[4] 对于heatmaps提取固定数量的keypoints. K top-left & K bottom-right

#### Step1.2: 完成keypoints的配对

[1] 每个valid pair of keypoints 定义了 object proposal. 
[2] 此处为了尽可能提高召回，不选择用cornernet的方式来处理；而是，只要top-left point和bottom-right point属于同一类物体，而且top-left point在bottom-right point的左上角，就认为他们属于同一个物体

### D.2.2 Step2: 利用two-step classification 进行proposals过滤

[1]用一个轻量的分类器，删除80%的proposal

[2] 之后使用一个refine classifier 对剩余的proposals判断class

#### Step2.1: 使用filter判断 Two-step classification for filtering proposals
![image](https://user-images.githubusercontent.com/26115141/110132939-a8184b00-7e06-11eb-94f6-41d3994f0382.png)

首先， 我们采用RoIAlign, 使用1个7x7的kernel来提取每个proposal的特征

之后，一个 32x7x7的conv layer来获取每个proposal的分类score. 

确实实现了 基于size/距离的筛选，

因为，IoUm其实就考虑了距离和size 从而计算正样本

#### Step2.2: 使用一个精修的classifier实现分类
#### Step2.2


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




