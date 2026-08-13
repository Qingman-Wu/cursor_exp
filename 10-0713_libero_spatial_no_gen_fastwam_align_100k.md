# 10-0713_libero_spatial_no_gen_fastwam_align_100k

## Material Passport

```text
artifact_type=experiment_result
experiment_id=10
status=completed
verification_status=ANALYZED_FROM_TRAINING_AND_EVALUATION_RECORDS
recorded_at=2026-08-13
```

## 状态

```text
训练完成
step 80000 / 90000 / 100000 正式评测完成并已同步
```

## 实验身份

```text
JOB_NAME=10-0713_libero_spatial_no_gen_fastwam_align_100k
start_time=2026-07-13 23:10 UTC+8
branch=egowam-stable
code_commit=be55a4450a8d2abf9d52c74bac264c3422f7499d
commit_subject=Align EgoWAM data and rollout with FastWAM
launcher=training/scripts/train_native_libero_spatial_no_gen_fastwam_align_100k.sh
SwanLab=https://swanlab.cn/@woovine/EgoWAM-Native/runs/dudp2hxy
```

## 实验目标

以实验 5 为基线，将数据和控制协议整体对齐 FastWAM，观察 Spatial 闭环成功率是否
从 88.0% 进一步提升。

这是组合消融，不用于区分每个变量的独立贡献。

## FastWAM-aligned contract

本实验整体替换数据与 rollout contract，具体包括：

```text
dataset_root=/root/wuqingman/datasets/nvidia/LIBERO_LeRobot_v3/libero_spatial
dataset_format=LeRobot v3
observation_image_keys=image,wrist_image
num_images_expected=2
frame_sampling_stride=1
action_horizon=32
action_exec_horizon=10
drop_incomplete_action_chunks=false
delta_action_dim_mask=true,true,true,true,true,true,false
normalization=min/max
normalization_eps=1e-8
```

训练在 run 目录生成 FastWAM-compatible `dataset_stats.json`，评测必须使用同一文件。

## 相对实验 5 的变化

```text
action horizon: 16 → 32
replan: 4（训练配置）→ 10
frame sampling stride: 4 → 1
episode tail: drop → padding + mask
padding delta action: first 6 dims=0
padding gripper: repeat last value
normalization: z-score → min/max
normalization eps: 1e-6 → 1e-8
normalization clip: none → [-5,5]
steps: 50k → 100k
```

保持：

```text
只用 libero_spatial
No-Gen
Understanding + Action full train
Action random init
双视角水平拼接
8D state
hidden size=1024
direct v-prediction
batch/GPU=32
GPUs=8
global batch=256
lr=1e-4
context_attention_mode=isolated
```

## 数据

```text
dataset=nvidia/LIBERO_LeRobot_v3/libero_spatial
effective samples=52,970
frame_sampling_stride=1
action_horizon=32
action_exec_horizon=10
drop_incomplete_action_chunks=false
```

Delta mask：

```text
true,true,true,true,true,true,false
```

即前 6 维为 delta action，gripper 为绝对/离散状态。

## 模型

```text
Understanding=full train
Generation=disabled
Action=full train
LM CE=disabled
Action flow loss=1
Image loss=0
lang_hidden_size=1024
act_hidden_size=1024
robot_state_dim=8
state injection=und
```

## Action Flow

```text
direct v-prediction
target=noise-action
shift=5
num_train_timesteps=1000
```

## Normalization

```text
mode=min/max
eps=1e-8
clip=[-5,5]
```

Run 目录生成：

```text
dataset_stats.json
```

评测必须使用该文件，不能再用实验 5 的 z-score `meta/stats.json`。

## 优化

```text
GPUs=8
batch/GPU=32
global batch=256
steps=100,000
lr=1e-4
optimizer=AdamW
betas=(0.9,0.95)
weight_decay=0
bf16
FSDP FULL_SHARD
save_steps=5000
checkpoint pruning=disabled
```

## 启动验证

```text
29 contract tests passed
8/8 distributed workers started
dataset_stats.json generated
step 10 loss=5.2069, grad norm=13.63
step 20 loss=1.0860, grad norm=3.31
step 50 loss=0.4403, grad norm=1.74
```

启动阶段无：

```text
OOM
Traceback
RuntimeError
NCCL error
```

## 训练结果

```text
100000/100000 完成
final loss≈0.0027
无 OOM / Traceback
```

最终 checkpoint：

```text
/root/wuqingman/RUN/
10-0713_libero_spatial_no_gen_fastwam_align_100k/
step_100000
```

已存在中间合并 checkpoint：

```text
step_70000_hf
step_80000_hf
step_90000_hf
step_100000_hf
```

## 正式评测协议

```text
suite=libero_spatial
MuJoCo=3.3.2
10 tasks × 50 trials
action_horizon=32
replan=10
num_inference_steps=10
wait_steps=30
normalization=min/max
stats=<run>/dataset_stats.json
context_attention_mode=isolated
flow seed 必须记录
```

主要对照：

```text
实验 5 replan=10: 440/500 = 88.0%
FastWAM spatial: 约 98.2%
```

## 80k / 90k / 100k 正式评测结果

三组评测同时运行；每个 checkpoint 都使用 GPU 0–7，协议完全相同：

```text
step_80000:  444/500 = 88.8%
step_90000:  454/500 = 90.8%  best
step_100000: 435/500 = 87.0%
```

此前已完成的中间 checkpoint 评测：

```text
step_50000: 431/500 = 86.2%
step_70000: 413/500 = 82.6%
```

因此当前已知结果覆盖 50k、70k、80k、90k、100k 五个 checkpoint；其中 90k 为最佳。

相对实验 5 `440/500 = 88.0%`：

```text
step_80000:  +0.8 percentage points
step_90000:  +2.8 percentage points
step_100000: -1.0 percentage points
```

正式评测使用 `wait_steps=30`，与 FastWAM
`configs/sim_libero.yaml:num_steps_wait=30` 一致。CLI `seed=0` 只固定 simulator；
action-flow noise 仍为未显式播种的 `torch.randn`。

完整逐任务结果：

```text
/root/wuqingman/RUN/10-0713_libero_spatial_no_gen_fastwam_align_100k/
EVALUATION_80K_90K_100K.md
```

## 评测交接

Run 目录已有：

```text
EVAL_HANDOFF.md
run_eval_exp10_parallel.sh
TRAIN_LAUNCHER.sh
CODE_COMMIT.txt
SHA256SUMS
```

评测输出：

```text
eval/libero_step50000_spatial_50trials_wait30/summary.csv
eval/libero_step70000_spatial_50trials_wait30/summary.csv
eval/libero_step80000_spatial_50trials_wait30/summary.csv
eval/libero_step90000_spatial_50trials_wait30/summary.csv
eval/libero_step100000_spatial_50trials_wait30/summary.csv
```

## 后续

1. 以 step 90000 的 90.8% 作为当前 Experiment 10 最优 checkpoint。
2. 若需要评测方差，使用相同 init-state 顺序重复多个 action-flow seed。
3. 继续拆分 horizon/stride/padding/min-max normalization 各自贡献。
