# 百度aistudio 2020 事件抽取赛道

------



依赖包：主要是tensorflow 1.12.0,另外使用了bojone的bert4keras，详见https://github.com/bojone/bert4keras

，其余见requirements.txt

目前主要集中使用机器阅读理解的方式来尝试解决事件抽取任务。主要分为两个阶段：

1、事件类型抽取

2、事件论元抽取，使用MRC的方式来做。

具体内容见知乎文章。

项目主体来自于本人另一个repo，使用MRC做实体识别，具体可参考https://github.com/qiufengyuyi/sequence_tagging

最终使用Retro-reader方法，在test1.json上的分数为0.856，使用了roberta-large-wwm.

对于roberta-wwm-base,分数为0.851.

## 生成k-fold训练数据：

根据不同阶段，生成两个阶段的k-fold训练数据，具体可参考gen_kfold_data.py


## 事件类型抽取：

```shell
bash run_event_classification.sh
```

## baseline事件论元抽取：

```shell
bash run_event_role.sh
```

## RetroReader-EAV问题是否可回答模块：

```shell
bash run_retro_eav.sh
```

## RetroReader-精读模块：

```shell
bash run_retro_rolemrc.sh
```
