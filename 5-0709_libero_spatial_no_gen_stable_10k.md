# 5-0709_libero_spatial_no_gen_stable_10k

## 状态

```text
中止 / 未完成
```

原计划训练 10,000 steps，实际日志只运行到：

```text
step=724/10000
latest loss=0.4378
```

没有完成 checkpoint，后续被同设置的正式 50k 实验替代：

```text
5-0709_libero_spatial_no_gen_stable_50k
```

## 实验目标

验证 LIBERO Spatial 上 No-Gen Action-only 训练能否稳定收敛。

## 数据

```text
dataset=nvidia/LIBERO_LeRobot_v3/libero_spatial
format=LeRobot v3
observation views=image,wrist_image
frame_sampling_stride=4
action_horizon=16
drop_incomplete_action_chunks=true
```

## 模型

```text
Understanding=full train
Generation=disabled
Action=full train, random init
LM CE=disabled
image loss=0
action flow loss=1
lang_hidden_size=1024
act_hidden_size=1024
robot_state_dim=8
state history=1
state injection=und
```

## Normalization

```text
action/state z-score
stats=libero_spatial/meta/stats.json
eps=1e-6
```

## 优化

```text
GPUs=8
batch/GPU=32
global batch=256
lr=1e-4
optimizer=AdamW
precision=bf16
FSDP=FULL_SHARD
```

## 代码

```text
branch=egowam-stable
experiment family commit≈515464a
```

## 日志

```text
/root/wuqingman/logs/
5-0709_libero_spatial_no_gen_stable_10k.log
```

## 结论

该实验不是有效的 10k 结果，仅是正式 50k 实验前的早期尝试。后续分析和评测均应使用：

```text
5-0709_libero_spatial_no_gen_stable_50k
```
