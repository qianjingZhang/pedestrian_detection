Corner Proposal Network for anchor-free, two-stage object detection

A. Abstract



# B. Introduction

## 分析 or 引言

### 目标定义

尽可能多的找到目标物体 + 尽可能准确的分配标签 高recall + 高precision

### 相关方法分析

| 目标 | 相关方法 ｜ 初步结论｜
｜ --- ｜ --- ｜ --- ｜
｜提高物体Recall| anchor-free vs anchor-based| anchor-based受限于anchor设定，对于常规物体有较好表现；但是，对于特殊size or shape的物体，召回能力不如anchor-free|
|提高物体precision| one-stage vs two-stage| two-stage方法引入类似多层筛选/回归方式，并且roi pooling可以更好滴完成feature和proposal的assign, 因此效果更佳|

### 网络设计

｜ 环节｜ 参考方法｜ 亮点 or 改进|
| --- | --- | ---|
| Proposal/RPN| Anchor-free中cornerNet| 不进行keypoints的配对，而是选择罗列所有可能的配对｜






C. Related Work

D. Methods


E. Ablation Study

F. Discussion
