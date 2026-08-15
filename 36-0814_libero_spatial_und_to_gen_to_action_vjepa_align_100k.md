# 36-0814_libero_spatial_und_to_gen_to_action_vjepa_align_100k

## Material Passport

```text
artifact_type=experiment_design
experiment_id=36
status=formal_training_running
verification_status=FORMAL_H20_TRAINING_STARTED_AND_FIRST_STEPS_VERIFIED
recorded_at=2026-08-14
last_verified_at=2026-08-16
```

## 实验定义

```text
baseline=Experiment 30
treatment=Experiment 30 + frozen V-JEPA Generation alignment
dataset=LIBERO Spatial
target_steps=100000
```

## 研究问题与假设

Exp30 已用 `Und -> Gen -> Action` attention 和 persistent joint denoising，让 Action 在闭环推理中
持续读取 Generation latent。Exp36 要回答：在不提供真实 future image 的前提下，用 frozen V-JEPA
future-keyframe 表征监督 Generation 中间层，能否改善 Generation 表征，并经 Gen→Action 条件路径提高
LIBERO Spatial 闭环成功率。

核心假设是 V-JEPA REPA 先直接约束 Generation，而不是直接约束 Action。与 Exp27 不同，Exp36 的
Generation 不读取 Action，因此不会形成 `Action -> Gen alignment loss` 的反向条件路径。

## 唯一实验处理

实验 36 保持 Exp30 的 `Und -> Gen -> Action` 层级 attention、persistent joint denoising、
1024/1024/1024 hidden、t+32 future keyframe、optimizer、batch 和评测协议，仅增加 Exp27
的 frozen V-JEPA layer-8 patch-token alignment。

```text
enable_vjepa_align=true
vjepa_align_coef=0.05
vjepa_align_warmup_steps=5000
vjepa_align_layer=8
vjepa_target_mode=future_keyframe
vjepa_target_pool=none
vjepa_input_size=384x384
vjepa_feature_dim=1024
vjepa_projector_hidden_dim=1536
vjepa_cache_format=egowam_exp27_v1
```

V-JEPA 只在训练期使用，评测不加载 teacher。由于 Exp30 中 Gen 不读取 Action，alignment
主要先优化 Generation 表征，再通过 Gen→Action 条件间接影响 Action。

训练损失为：

```text
L_total = L_image + L_action + warmup(lambda_vjepa) * L_vjepa
lambda_vjepa: step 0–5000 从 0 线性增加到 0.05，之后保持 0.05
```

## 可复现实验配置

```text
code_repository=https://github.com/Zheng-Chong/Ego-WAM.git
code_branch=egowam-stable
exp36_implementation_commit=2db9b526e0c220ed2d4324057cda4b75428e02fe
exp30_baseline_commit=df97591b9eff70f9c015cd555e475f3cebed5f4a
future_target=t+32 future keyframe
dataset=nvidia/LIBERO_LeRobot_v3/libero_spatial
effective_samples=52,970
frame_sampling_stride=1
normalization=min/max, eps=1e-8, clip=[-5,5]
action_horizon=32
action_exec_horizon=10
batch/GPU=32
GPUs=8
global_batch=256
gradient_accumulation=1
optimizer=AdamW, lr=1e-4, weight_decay=0.01
schedule=cosine_with_min_lr, warmup_ratio=0.05, min_lr_ratio=0.01
target_steps=100000
save_steps=5000
action_initialization=random (init_action_from_gen=false)
SwanLab=https://swanlab.cn/@woovine/EgoWAM-Native/runs/2wzu3ksl
formal_checkpoint=not produced
formal_training=running
formal_evaluation=not started
```

正式训练必须 checkout 上述 Exp36 完整 commit SHA。分支名只用于定位，不能替代 commit；如果训练时
`git rev-parse HEAD` 与记录不同，必须保存实际 SHA 和相对差异，不能沿用本实验结果登记。

## 代码与入口

```text
training launcher=training/scripts/train_native_libero_spatial_exp36_vjepa_joint_100k.sh
single-checkpoint eval=training/scripts/eval_native_libero_spatial_exp36.sh
batch eval=training/scripts/eval_native_libero_spatial_exp36_checkpoints.sh
implementation design=training/EXPERIMENT_36_LIBERO_SPATIAL_VJEPA_JOINT_CN.md
V-JEPA teacher=/home/work/wuqingman/Ego-WAM/models/vjepa2_1_vitl_dist_vitG_384.pt
V-JEPA cache=${WORKSPACE_ROOT}/datasets/vjepa_cache/libero_spatial
```

评测端固定：

```text
suite=libero_spatial
tasks=10
trials/task=50
episodes/checkpoint=500
checkpoints=5k,10k,20k,30k,40k,50k,60k,70k,80k,90k,100k
num_inference_steps=10
num_wait_steps=5
generation_image_source=none
joint_generation_action_sampling=true
gen_action_attention_mode=und_to_gen_to_action
```

评测模型明确关闭 `enable_vjepa_align`。即使 checkpoint 包含 V-JEPA projector 参数，backend 也只加载
闭环推理所需参数，不加载 teacher 或 cache。

## 验收与记录要求

1. 启动日志中的非 V-JEPA 配置必须与 Exp30 一致。
2. 训练前验证 V-JEPA cache metadata、patch grid、feature dimension、camera key 和 frame index。
3. 记录 image loss、action loss、原始/加权 alignment loss、LR、grad norm 和 warmup coefficient。
4. 每个正式 checkpoint 记录 SHA256、SwanLab run URL 和 500-episode 成功率。
5. 主比较为相同步数下 Exp36 与 Exp30 的 LIBERO Spatial 成功率；smoke 指标不参与性能结论。

## 验证记录

```text
rtx5090-180 isolated checkout=/root/codex-egowam-exp36-20260814
Exp36 targeted behavioral tests=5 passed
full training/tests suite=72 passed
CUDA smoke=passed on NVIDIA GeForce RTX 5090
total_loss=0.640642
vjepa_align_loss_training_forward=0.859716
gradient_norm=7.279457
```

真实 V-JEPA teacher checkpoint 已在该机确认存在；Exp27 全量 cache、U1 base model 和可直接加载的
Exp20/30 checkpoint 未找到。因此 CUDA smoke 只覆盖 tiny model 的 joint + alignment 数值路径，不构成
真实数据、真实 8B checkpoint 或正式 LIBERO 闭环结果。

## 2026-08-16 正式训练启动记录

```text
machine=Alibaba Cloud DSW, 8 x NVIDIA H20 96GB
formal_start_time=2026-08-16 00:34:19 UTC+8
tmux_session=exp36_pipeline
code_checkout=/root/wuqingman/Ego-WAM-exp36-filtered
code_commit=2db9b526e0c220ed2d4324057cda4b75428e02fe
code_status=clean detached HEAD
JEPA-WAM_commit=8452205c49c3a90df6c3c46e184fd6fa38890e70
V-JEPA2_commit=204698b45b3712590f06245fbfba32d3be539812
teacher_sha256=7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6
cache=/root/wuqingman/datasets/vjepa_cache/libero_spatial
cache_shards=432/432
cache_size=117GB
SwanLab_run=https://swanlab.cn/@woovine/EgoWAM-Native/runs/2wzu3ksl
output=/root/wuqingman/RUN/36-0814_libero_spatial_und_to_gen_to_action_vjepa_align_100k
log=/root/wuqingman/logs/36-0814_libero_spatial_und_to_gen_to_action_vjepa_align_100k.log
cache_log=/root/wuqingman/logs/36-0814_libero_spatial_und_to_gen_to_action_vjepa_align_100k_vjepa_cache.log
```

启动日志已确认 `und_to_gen_to_action`、isolated Understanding、1024/1024/1024、
AdamW、batch/GPU=32、cosine-with-min-LR、5000-step warmup，以及与 Exp27 相同的
layer-8、future-keyframe、patch-token、coef=0.05 V-JEPA alignment contract。模型为
2.700B trainable parameters，其中 V-JEPA projector 3.15M。8 卡约占用 59GB/GPU，
正式训练和 SwanLab 云端记录均已进入有效 step。

严格来说，Exp36 和 Exp27 的 V-JEPA/REPA 实现及训练目标相同，训练侧主要 treatment 是
attention topology；但闭环评测时 Exp36 还继承 Exp30 的 persistent joint
Generation/Action solver 和 `generation_image_source=none`。因此最终 Exp36-vs-Exp27
闭环差异不能全部归因于 attention mask，除非补充匹配推理 solver 的独立消融。

## 待办

```text
monitor 8-GPU formal training through the first 5000 steps
record SwanLab run and final loss/LR/grad norm
record checkpoint list and SHA256
run 10-task × 50-trial closed-loop evaluation
compare Exp36 against Exp30 at matched checkpoints
```
