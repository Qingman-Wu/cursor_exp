# 实验 14：LIBERO Spatial No-Gen FastWAM Optimizer Alignment 100k

## 实验身份

```text
JOB_NAME=14-0723_libero_spatial_no_gen_fastwam_optim_align_100k
baseline=10-0713_libero_spatial_no_gen_fastwam_align_100k
branch=egowam-stable
code_commit=8898f9a
launcher=training/scripts/train_native_libero_spatial_no_gen_fastwam_optim_align_100k.sh
status=正式训练尚未启动；8 卡 smoke 已通过
```

## 目标与对照

实验 14 以实验 10 为严格基线，保持模型、数据和控制 contract 不变，只对齐 FastWAM
公开 LIBERO 配置中的优化器 schedule、weight decay 和 activation checkpointing。

本实验不修改 batch size，不修改图像预处理，也不引入 Generation branch。

## 相对实验 10 的变量

```text
peak learning rate             1e-4（不变）
LR schedule                    constant -> 5% warmup + cosine decay
warmup                         1000 steps -> 5000 steps
minimum LR                     不适用 -> 1e-6
AdamW weight decay             0 -> 0.01
gradient checkpointing         true -> false
```

实现参数：

```text
lr=1e-4
lr_scheduler_type=cosine_with_min_lr
init_steps=0
warmup_ratio=0.05
min_lr_ratio=0.01
weight_decay=0.01
gradient_checkpointing=false
```

100,000 optimizer updates 下：

```text
前 5,000 steps：linear warmup 到 1e-4
后 95,000 steps：cosine decay，从 1e-4 降至 1e-6
```

## 数据

```text
dataset=nvidia/LIBERO_LeRobot_v3/libero_spatial
format=LeRobot v3
effective samples=52,970
frame_sampling_stride=1
action_horizon=32
action_exec_horizon=10
drop_incomplete_action_chunks=false
delta_action_dim_mask=true,true,true,true,true,true,false
```

双视角：

```text
[image | wrist_image]
每路 256×256
拼接后 512×256
ImageNet mean/std normalization
```

## 模型

```text
Understanding: full train
Generation: disabled / not built
Action: full train
Value: disabled
LM CE: disabled
Action loss weight: 1
Action initialization: random
context_attention_mode=isolated
lang_hidden_size=1024
act_hidden_size=1024
robot_state_dim=8
robot_state_num_tokens=1
robot_state_injection_branch=und
```

总参数量和可训练参数量：

```text
total=1,835.62M
trainable=1,835.62M
Understanding=997.43M
Action=838.19M
```

## Normalization 与 Flow

```text
normalize_action_state=true
action_state_norm_mode=min/max
action_state_norm_eps=1e-8
normalization clip=[-5,5]

action_train_shift=5
action_infer_shift=5
action_num_train_timesteps=1000
target_v=noise-action
```

## 优化和运行配置

```text
GPUs=8
batch/GPU=32
gradient_accumulation=1
global_batch=256
total_steps=100000
optimizer=AdamW
betas=(0.9,0.95)
eps=1e-8
weight_decay=0.01
gradient_clip=1.0
precision=bf16
FSDP=FULL_SHARD
gradient_checkpointing=false
save_steps=5000
max_keep_ckpt=0
```

## 8 卡 Smoke

Smoke job：

```text
14-exp14_fastwam_optim_smoke
```

结果：

```text
8×H20
batch/GPU=32
global_batch=256
20/20 steps completed
step_20 checkpoint saved
无 OOM
无 Traceback
无 NCCL error
```

Smoke 输出：

```text
/root/wuqingman/RUN/14-exp14_fastwam_optim_smoke
```

该 smoke 仅验证配置、显存和分布式训练链路，不作为正式训练结果。

## 正式启动命令

```bash
cd /root/wuqingman/Ego-WAM-egowam-stable/training

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NCCL_DEBUG=WARN \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
TOKENIZERS_PARALLELISM=false \
MAIN_PROCESS_PORT=29695 \
bash scripts/train_native_libero_spatial_no_gen_fastwam_optim_align_100k.sh \
2>&1 | tee /root/wuqingman/logs/14-0723_libero_spatial_no_gen_fastwam_optim_align_100k.log
```

## 正式评测协议

主结果必须沿用实验 10 的评测协议：

```text
suite=libero_spatial
MuJoCo=3.3.2
10 tasks × 50 trials
action_horizon=32
action_exec_horizon/replan=10
num_inference_steps=10
num_wait_steps=5
action_state_norm_mode=min/max
action_state_stats_path=<run>/dataset_stats.json
context_attention_mode=isolated
```

可额外报告 FastWAM 风格 `num_wait_steps=30`，但不能替代 wait=5 的主对照结果。

## 当前状态与待办

```text
code committed and pushed: 8898f9a
8-card smoke: passed
formal training: pending
SwanLab: pending
formal checkpoints: pending
formal evaluation: pending
```
