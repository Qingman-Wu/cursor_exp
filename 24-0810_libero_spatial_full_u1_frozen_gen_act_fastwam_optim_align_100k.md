# 24-0810_libero_spatial_full_u1_frozen_gen_act_fastwam_optim_align_100k

## Material Passport

```text
artifact_type=experiment_result
experiment_id=24
status=training_configuration_recorded_results_pending
verification_status=ANALYZED_FROM_PROVIDED_TRAINING_MATERIAL
recorded_at=2026-08-13
```

## 状态与身份

```text
baseline=Experiment 20
JOB_NAME=24-0810_libero_spatial_full_u1_frozen_gen_act_fastwam_optim_align_100k
repo=/root/wuqingman/Ego-WAM-egowam-stable
code_commit=4e4d2639c7b9e765af23b47d4589062b8bf56af4
launcher=training/scripts/train_native_libero_spatial_full_u1_frozen_gen_act_fastwam_optim_align_100k.sh
target_steps=100000
training/evaluation results=待补
```

## 研究问题

保留 SenseNova-U1 原始全宽 Understanding 和 Generation 表示并完全冻结，只优化随机初始化的
Action expert，同时保留未来图像 Generation 辅助目标，让 image flow loss 是否能通过冻结的
Generation computation graph 监督 Action。

## 相对实验 20 的 treatment bundle

```text
Und width: 1024 compressed -> original U1 4096
Gen width: 1024 compressed -> original U1 4096
Und train mode: full -> frozen
Gen train mode: full -> frozen
Action: random initialized, trainable（保持）
Generation auxiliary image target: 保留
```

这同时改变宽度、冻结策略和可训练参数分布，不是单独的 freeze ablation。

## 冻结 Generation 辅助梯度

仅设置 `gen_train_mode=frozen` 会让旧逻辑关闭 Generation 数据和 image loss。Exp24 额外设置：

```text
enable_frozen_gen_aux_loss=true
disable_gen_forward=false
image_gen_loss_coef=1.0
```

数据请求：

```text
und=false, gen=true, act=true, value=false
```

梯度路径：

```text
Action hidden -> trainable Action K/V -> frozen Generation attention
-> frozen Generation flow head -> image flow loss -> Action gradients
```

Generation 参数 `requires_grad=false` 且不进入 optimizer，但 forward 不能包在 `detach()` 或
`no_grad()` 中，否则 image loss 无法对 Action 输入求梯度。验收时必须检查 Generation 参数无
梯度、Action 参数同时接收到 action loss 与 image auxiliary loss 的梯度。

## 数据、模型与训练配置

```text
dataset=/root/wuqingman/datasets/nvidia/LIBERO_LeRobot_v3/libero_spatial
base_model=/root/wuqingman/models/SenseNova-U1-8B-MoT-SFT
venv=/root/wuqingman/venvs/egowam-py311
Und/Gen width=4096/4096
Action=random initialized, trainable
GPUs=8
batch/GPU=32
global_batch=256
gradient_accumulation=1
total_steps=100000
logging_steps=10
save_steps=5000
max_keep_ckpt=0
gradient_checkpointing=true
```

未在当前材料中给出的 optimizer、attention mask、future horizon、normalization 和 loss 系数，
应从 commit `4e4d2639` 的 launcher 或实际训练日志确认，不能仅按 Exp20 自动推断。

## 启动方式

```bash
cd /root/wuqingman/Ego-WAM-egowam-stable
source /root/wuqingman/venvs/egowam-py311/bin/activate
export WORKSPACE_ROOT=/root/wuqingman
export MODEL_NAME_OR_PATH=/root/wuqingman/models/SenseNova-U1-8B-MoT-SFT
export TOKENIZER_PATH="$MODEL_NAME_OR_PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MAIN_PROCESS_PORT=29697
export JOB_NAME=24-0810_libero_spatial_full_u1_frozen_gen_act_fastwam_optim_align_100k
export OUTPUT_DIR="/root/wuqingman/RUN/${JOB_NAME}"
export total_steps=100000 batch_size=32 grad_accm=1
export logging_steps=10 save_steps=5000 max_keep_ckpt=0
export gradient_checkpointing=true
export use_swanlab=true swanlab_experiment="$JOB_NAME"
unset RESUME_FROM

nohup bash training/scripts/train_native_libero_spatial_full_u1_frozen_gen_act_fastwam_optim_align_100k.sh \
  > "/root/wuqingman/logs/${JOB_NAME}.log" 2>&1 &
```

## 验收与待补

必须确认：Und/Gen 实际为 4096；两者 `requires_grad=false`；Action trainable；Generation forward
未禁用；image loss 非零且能对 Action 反传；optimizer 不包含冻结参数。

待补：训练开始/结束时间、SwanLab、参数量、最终 loss/LR/grad norm、checkpoint SHA256、
显存与吞吐、正式 10-task x 50-trial 闭环结果。
