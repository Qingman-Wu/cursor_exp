# 6-0710_libero_all_no_gen_stable_100k

## 状态

```text
主动中止
```

运行到：

```text
step≈11,364/100,000
latest loss≈0.2847
```

已保存：

```text
step_5000
step_10000
```

## 实验目标

将实验 5 的 No-Gen Action-only 配置扩展到 NVIDIA LIBERO LeRobot v3 根目录下的全部
suite。

## 数据

扫描到五套：

```text
libero_spatial
libero_object
libero_goal
libero_10
libero_90
```

经过 stride=4、action horizon=16 过滤后：

```text
libero_10       24,095
libero_90      129,096
libero_goal     11,568
libero_object   15,217
libero_spatial  11,792
total          191,768
```

`libero_90` 占约 67%，会主导训练分布。

## 模型

```text
Understanding + Action
Generation disabled
Action random init
total/trainable params=1,835.62M
```

## 核心设置

```text
action_horizon=16
action_exec_horizon=4
frame_sampling_stride=4
state_dim=8
state_history=1
normalization=per-suite z-score
batch/GPU=32
GPUs=8
global batch=256
steps=100,000
lr=1e-4
bf16
FSDP FULL_SHARD
```

## 代码

```text
branch=egowam-stable
launcher commit≈a059b79
launcher=training/scripts/train_native_libero_all_no_gen.sh
```

## 中止原因

主流 VLA 的标准 LIBERO benchmark 使用：

```text
Spatial
Object
Goal
Long/libero_10
```

通常不将 `libero_90` 混入联合训练。`libero_90` 属于原始 lifelong-learning 预训练协议。

继续该实验会使：

- 约三分之二 batch 来自 `libero_90`；
- 训练目标与 FastWAM/π0.5 等四套标准协议不一致；
- 难以解释与实验 5 的对照。

因此主动停止，并用以下正式实验替代：

```text
6-0711_libero_4suite_no_gen_stable_100k
```

## 输出

```text
/root/wuqingman/RUN/6-0710_libero_all_no_gen_stable_100k
/root/wuqingman/logs/6-0710_libero_all_no_gen_stable_100k.log
```

## 结论

该实验用于确认数据扫描和分布问题，不作为正式四套 LIBERO 结果。
