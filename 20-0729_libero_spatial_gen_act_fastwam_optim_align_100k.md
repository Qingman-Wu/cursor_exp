# 20-0729_libero_spatial_gen_act_fastwam_optim_align_100k

## Material Passport

```text
artifact_type=experiment_result
experiment_id=20
status=training_configuration_recorded_results_pending
verification_status=ANALYZED_FROM_PROVIDED_TRAINING_MATERIAL
recorded_at=2026-08-13
```

## 状态

```text
训练设计、固定代码版本与启动方式已登记
训练完成、最终 loss、checkpoint 和正式闭环评测：待补
machine=wqm2
```

当前材料包含启动方式与 SwanLab 链接，但未提供训练结束日志、实际 step、checkpoint 清单或
`summary.csv`，因此不能将 100,000 目标步数写成已完成事实。

## 实验身份

```text
JOB_NAME=20-0729_libero_spatial_gen_act_fastwam_optim_align_100k
branch=egowam-stable
base_code_commit=f98d358d09f566f263461e1923e8f3a58faf74b8
experiment_code_commit=552753c75f43903f77b6ea5c9843c4f48bce1e89
launcher=training/scripts/train_native_libero_spatial_gen_act_fastwam_optim_align_100k.sh
dataset=LIBERO Spatial
target_steps=100000 optimizer updates
machine=wqm2
SwanLab=https://swanlab.cn/@woovine/EgoWAM-Native/runs/wjsyshkc/chart
```

建议使用固定 commit 的独立 worktree：

```bash
git worktree add --detach /root/wuqingman/Ego-WAM-exp20 \
  552753c75f43903f77b6ea5c9843c4f48bce1e89
```

## 研究问题

在保持实验 18 的 Spatial 数据、Action contract、optimizer 和训练长度不变的条件下，加入
从 SenseNova-U1 初始化并压缩至 1024 hidden size 的 Generation expert，联合预测未来图像。

## 相对实验 18 的 treatment bundle

实验 20 的变化不是只新增一个 loss 标量，而是以下耦合处理：

```text
构建并全参数训练 Generation expert
Generation 从 SenseNova-U1 初始化，并由 4096 压缩至 1024
加入 future-image target
加入 image flow loss
使用 Generation query 可见性设计
```

因此若结果改变，归因对象是整个 Generation co-training bundle，不是单个 image loss 系数。

## 模型、初始化与 attention

```text
Understanding=SenseNova-U1 initialized, compressed to 1024, full train
Generation=SenseNova-U1 initialized, compressed to 1024, full train
Action=random initialized, full train
context_attention_mode=isolated
gen_action_attention_mode=gen_reads_und_action
```

`gen_reads_und_action` 的语义为 Generation query 可读取全部 Understanding 和 Action token。
该设置不表示 Action query 可读取 Generation token；Action 维持不读取 Gen 的设计。因此 Exp20
不属于 later joint Generation-Action sampling / `action_reads_gen` 的结构。

## 继承的 Exp18 contract

```text
suite=libero_spatial
views=image,wrist_image
frame_sampling_stride=1
action_horizon=32
action_exec_horizon=10
drop_incomplete_action_chunks=false
delta_action_dim_mask=true,true,true,true,true,true,false
robot_state_dim=8
robot_state_injection_branch=und
action flow=direct v-prediction
shift=5
target_v=noise-action
normalization=min/max
normalization_eps=1e-8
GPUs=8
batch/GPU=32
global_batch=256
gradient_accumulation=1
lr=1e-4
optimizer=AdamW
lr_scheduler=cosine_with_min_lr
warmup_ratio=0.05
min_lr_ratio=0.01
weight_decay=0.01
gradient_checkpointing=false
dtype=bf16
save_steps=5000
max_keep_ckpt=0
```

这些继承项仍需以后续实际启动日志和训练配置快照验证。

## 启动方式

```bash
cd /root/wuqingman/Ego-WAM-exp20
source /root/wuqingman/venvs/egowam-py311/bin/activate

export WORKSPACE_ROOT=/root/wuqingman
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MAIN_PROCESS_PORT=29695
export JOB_NAME=20-0729_libero_spatial_gen_act_fastwam_optim_align_100k
export OUTPUT_DIR="/root/wuqingman/RUN/${JOB_NAME}"
export swanlab_experiment="${JOB_NAME}"

bash training/scripts/train_native_libero_spatial_gen_act_fastwam_optim_align_100k.sh \
  2>&1 | tee "/root/wuqingman/logs/${JOB_NAME}.log"
```

## 正式评测要求

评测必须构建与 checkpoint 一致的 Generation expert。闭环推理是否禁用 Generation forward、
使用的 checkpoint、wait steps、action-flow seed、reset protocol 和每 task trials 数尚未在当前
材料中给出，不能从 Exp18/22 自动继承。

## 待补材料

1. 训练开始/结束时间、实际退出码、最终 step、loss、LR 和 grad norm。
2. checkpoint 列表、SHA256、`dataset_stats.json` 和是否成功导出 HF 权重。
3. future-image horizon、image flow parameterization 与 image/action loss 系数。
4. 正式评测脚本、模型构造参数、action-flow seed 和 50 trials/task `summary.csv`。
5. 同 Exp18 checkpoint/评测协议下的 paired comparison。
