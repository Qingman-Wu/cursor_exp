# 6-0711_libero_4suite_no_gen_stable_100k

## 状态

```text
训练完成
四套 Step85000 正式评测完成：1436/2000，71.8%
```

## 实验身份

```text
JOB_NAME=6-0711_libero_4suite_no_gen_stable_100k
branch=egowam-stable
code_commit=8159aa38cb2f3159b7a13a90c5d033db367c64d3
launcher=training/scripts/train_native_libero_4suite_no_gen.sh
SwanLab=https://swanlab.cn/@woovine/EgoWAM-Native/runs/g0ysh1lx
```

## 实验目标

将实验 5 的 No-Gen Action-only 训练扩展到标准四套 LIBERO，并排除 `libero_90`。

## 数据

```text
libero_10       24,095
libero_object   15,217
libero_spatial  11,792
libero_goal     11,568
total           62,672
```

明确排除：

```text
libero_90
```

数据配置：

```text
frame_sampling_stride=4
action_horizon=16
drop_incomplete_action_chunks=true
views=image,wrist_image
horizontal concat
state_dim=8
state_history=1
```

## 模型

```text
model=EgoWAMNativeModel
total/trainable params=1,835.62M
Understanding=997.43M, SenseNova init, full train
Generation=disabled
Action=838.19M, random init, full train
LM CE=disabled
Value=disabled
```

## Normalization

```text
mode=z-score
normalize_action_state=true
eps=1e-6
每个 suite 独立 meta/stats.json
```

## Action Flow

```text
direct v-prediction
shift=5
num_train_timesteps=1000
target=noise-action
```

## 优化

```text
GPUs=8
batch/GPU=32
global batch=256
steps=100,000
lr=1e-4
AdamW betas=(0.9,0.95)
weight_decay=0
clip_grad=1
bf16
FSDP FULL_SHARD
save_steps=5000
保留全部 checkpoint
```

## 训练结果

```text
100000/100000 完成
final action loss=0.0150
final grad norm=0.18
无 OOM / Traceback / RuntimeError
```

最终 checkpoint：

```text
/root/wuqingman/RUN/6-0711_libero_4suite_no_gen_stable_100k/
step_100000
```

完整实验记录：

```text
/root/wuqingman/RUN/6-0711_libero_4suite_no_gen_stable_100k/
EXPERIMENT.md
```

## Step85000 跨作业评测（实验 8）

已合并并备份：

```text
/mnt/public/wuqingman/RUN/
6-0711_libero_4suite_no_gen_stable_100k/
step_85000_hf/model.safetensors
```

四套 stats：

```text
/mnt/public/wuqingman/libero_eval/eval_stats/
```

评测环境：

```text
MuJoCo 3.3.2
EGL NVIDIA vendor
no-gen checkpoint structure
replan 需统一使用 10
50 trials/task
四套共 2000 episodes
```

已验证：

```text
spatial smoke=1/1
```

完整评测日志：

```text
/root/wuqingman/logs/eval_step85000_full.log
```

正式结果：

| Suite | Successes | Episodes | Success rate |
|---|---:|---:|---:|
| libero_spatial | 370 | 500 | 74.0% |
| libero_object | 453 | 500 | 90.6% |
| libero_goal | 344 | 500 | 68.8% |
| libero_10 | 269 | 500 | 53.8% |
| **Overall** | **1436** | **2000** | **71.8%** |

评测实验文档：[`8-eval_exp6_step85000.md`](./8-eval_exp6_step85000.md)

逐任务原始结果：

```text
results/8-eval_exp6_step85000/summary.csv
```

## 后续

1. 合并并备份 Step100000。
2. 使用相同评测协议评测 Step100000。
3. 与实验 7 Gen+Act 和实验 10 FastWAM-aligned 对比。
