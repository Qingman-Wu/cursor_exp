# 30-0812_libero_spatial_und_to_gen_to_action_joint_100k

## Material Passport

```text
artifact_type=experiment_result
experiment_id=30
status=implementation_smoke_verified_training_results_pending
verification_status=ANALYZED_FROM_DESIGN_AND_SMOKE_RECORDS
recorded_at=2026-08-13
```

## 实验身份

```text
JOB_NAME=30-0812_libero_spatial_und_to_gen_to_action_joint_100k
direct_baseline=Experiment 20
code_base_commit=92d343b7909522e8f06f96bc5406080eabe1b37f
uncommitted_experiment_diff=有（材料记录）
dataset=LIBERO Spatial
target_steps=100000
formal training/evaluation=待补
```

## 科学问题

当 Action query 能读取在线 Generation latent，且 Generation 与 Action 在同一 flow solver 中
持续联合去噪时，是否能让 Action 使用更接近未来状态的表征，改善 LIBERO Spatial 闭环动作质量。

## Treatment bundle

```text
Experiment 30 = Experiment 20
              + Action query reads Generation tokens
              + persistent Generation latent during inference
              + joint Generation/Action denoising solver
              - zero-image future-frame placeholder
```

这是不可拆分的 interaction-and-inference bundle。主结果不能声称收益来自“仅 mask”或“仅联合
采样”；需要独立消融才能拆分归因。

## Attention topology

```text
Query Und: original isolated rules; no Gen/Action
Query Gen: reads all valid Und tokens in the same document; no Action
Query Action: original observation/causal conditions + Generation tokens
```

逻辑路径：

```text
Und ----> Gen ----> Action
  \----------------> Action
```

`isolated` 仍只限制 Understanding 内部 modality 交互，不撤销 Gen 对同一 document 有效 Und
token 的读取。Gen 不读取 Action，避免未来图像由 Action 反向泄漏。

## 训练与推理对应关系

训练时对 future keyframe clean patch latent `x_future` 使用 Generation flow：

```text
z_gen(t)=t*x_future+(1-t)*epsilon_gen
v_gen=(x_future-z_gen(t))/max(1-t,eps)
```

Action 保留实验 20 的 shifted flow matching；第一版不强制 Gen/Action 共用 timestep，也不改变
image/action loss 权重。

推理时每个 action chunk：

```text
一次采样 generation_z_0 和 action_z_0
整个 solver 过程中持久更新两个 latent
每个 solver step 一次 model forward，同时得到 generation_pred/action_pred
最终只输出 action chunk，Gen latent 不被真实 future image 替换
```

这避免了训练读取真实 future RGB、推理读取 zero image 的条件分布差异，也避免同一 Action solver
内 Gen 条件反复重新采样。

## 继承实验 20 的配置

```text
Und/Gen/Act width=1024/1024/1024
Und/Gen/Act train mode=full/full/full
Action initialization=random; init_action_from_gen=false
future keyframe=t+32
image/action flow loss=1.0/1.0
context_attention_mode=isolated
optimizer=AdamW; lr=1e-4; warmup=5%; cosine_with_min_lr
min_lr_ratio=0.01; weight_decay=0.01
batch/GPU=32; GPUs=8; total_steps=100000; save_steps=5000
```

实验 30 不混入实验 21 的 4096-wide U1、实验 24 的冻结策略、实验 27 的 V-JEPA/REPA loss，
也不改变 optimizer/scheduler。

## 正式启动与评测

```text
train_launcher=training/scripts/train_native_libero_spatial_exp30_joint_100k.sh
eval_launcher=training/scripts/eval_native_libero_spatial_exp30.sh
batch_eval_launcher=training/scripts/eval_native_libero_spatial_exp30_checkpoints.sh
```

训练启动日志必须确认：

```text
gen_action_attention_mode=und_to_gen_to_action
enable_gen_branch=true
disable_gen_forward=false
include_future_keyframe=true
hidden sizes=1024/1024/1024
image_gen_loss_coef=1.0
action_gen_loss_coef=1.0
```

评测固定：

```text
suite=libero_spatial
10 tasks x 50 trials/checkpoint=500 episodes
action horizon/execution horizon=32/10
num_inference_steps=10; num_wait_steps=5
generation_image_source=none
GEN_ACTION_ATTENTION_MODE=und_to_gen_to_action
JOINT_GENERATION_ACTION_SAMPLING=true
dtype=bf16
normalization=dataset_stats.json min/max
```

主 checkpoint 曲线计划为 5k、10k、20k、30k、40k、50k、60k、70k、80k、90k、100k。必须
报告每 task、overall/500、checkpoint path、sim/flow seed 和 eval code version。不得用 Exp20
checkpoint 作为 Exp30 性能结果；Exp20 只能做接口 smoke。

## 已完成实现验证

```text
static checks=py_compile, bash -n, git diff --check passed
CPU contract tests=67 passed
RTX 5090 tiny CUDA smoke=train forward/backward + 2-step joint denoising passed
RTX 5090 real checkpoint smoke=Experiment 20 step 65000 loaded and joint forward passed
generation_pred shape=(1,1,3072)
action_pred shape=(1,16,7)
max CUDA allocation=5.29 GiB
```

这些只证明实现和接口可运行，不构成实验 30 的训练效果或成功率证据。

## 结果登记

```text
training status=待补
final loss/LR/grad norm=待补
checkpoint list/SHA256=待补
LIBERO closed-loop results=待补
```

## 验收与诊断

必须验证 5k 前无 NaN/Inf/OOM，固定 observation+flow seed 时 action chunk 可复现，10 个 task
均完成 50 trials，并记录 Gen/Action latent 是否有限、持续演化且对 observation 有响应。

若 image loss 快速下降但 action loss 或闭环无改善，不能宣称 Generation 辅助表征转化为控制收益。
针对低成功率，优先检查 checkpoint/stats、joint sampling 开关、generation image source、
normalization/action convention、reset/camera/physics，再做逐 task 失败分类。

## 后续消融

```text
30-A: 训练新 mask，评测禁止 Action 读 Gen
30-B: Gen/Action 共用或对齐 timestep
30-C: Gen 读取 Action（需重做防泄漏契约）
30-D: 冻结 Gen 或停止 Gen->Action 梯度
30-E: 叠加 V-JEPA/REPA（独立命名）
```
