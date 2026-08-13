# 27-0811_libero_spatial_vjepa_align_100k

## Material Passport

```text
artifact_type=experiment_result
experiment_id=27
status=smoke_verified_formal_training_results_pending
verification_status=ANALYZED_FROM_IMPLEMENTATION_AND_SMOKE_RECORDS
recorded_at=2026-08-13
```

## 实验身份

```text
display_name=27-0811_libero_spatial_vjepa_repa_100k
JOB_NAME=27-0811_libero_spatial_vjepa_align_100k
branch=egowam-stable
baseline=Experiment 20
code_commit=f544c29f5d544bc2a9cb0ac90c015c3337dcf9ab
dataset=LIBERO Spatial
target_steps=100000
formal_training/evaluation results=待补
```

## 研究问题与唯一 treatment

在实验 20 的 Generation+Action 联合训练基线上，引入 frozen V-JEPA visual teacher，将
Generation 第 8 个 Transformer layer 的 intermediate hidden 对齐到 t+32 未来帧的 V-JEPA
patch-token 表征。

```text
Experiment 27 = Experiment 20
              + frozen V-JEPA teacher
              + Generation hidden -> future V-JEPA feature alignment loss
```

第一版保留原有 image flow loss 和 action flow loss，避免把“增加表征对齐”和“删除像素目标”
两个变量混在一起。

## 配置

```text
enable_vjepa_align=true
vjepa_align_coef=0.05
vjepa_align_layer=8
vjepa_target_mode=future_keyframe
vjepa_target_stopgrad=true
target=t+32 future keyframe
loss=normalized cosine
warmup=first 5000 optimizer steps
image_gen_loss_coef=1.0
action_gen_loss_coef=1.0
gen_action_attention_mode=gen_reads_und_action
inference=不加载 V-JEPA
```

总损失：

```text
L_total = L_action + L_image + lambda_vjepa * L_vjepa
```

## Teacher 与梯度路径

```text
teacher_checkpoint=/home/work/wuqingman/Ego-WAM/models/vjepa2_1_vitl_dist_vitG_384.pt
teacher_arch=vjepa2_1_vit_large_384
teacher_output_dim=1024
teacher_tokens=final-layer patch tokens
teacher_mode=frozen, stop-gradient
```

V-JEPA 是训练期旁路，不进入 MoT token 序列，也不成为推理输入。`L_align` 作用于 Generation
hidden，因此可更新 Generation、Generation 读取到的 Understanding/Action 表示和 alignment
projector。若 detach Action K/V，alignment 将不能辅助 Action，必须作为独立消融。

## Feature cache 与双相机映射

两路相机分别由 V-JEPA 编码，不先拼 RGB：

```text
vjepa[camera_key]=[T,N_patch,D]
frame_index=[T]
metadata=arch,input_size,patch_grid,normalization,resize,camera_mode,checkpoint
```

训练侧 U1 Generation 将 `image,wrist_image` 沿宽度拼接，因此 target 按 camera patch grid 恢复
二维结构，再沿宽度拼接，并 bilinear interpolation 到 Generation grid。禁止按 flatten token
index 截断、repeat 或一维插值。Projector 为 `Linear(D_gen=1024,D_vjepa=1024)`。

全量 cache：

```text
cache_root=/root/wuqingman/datasets/vjepa_cache/libero_spatial
episode shards=432
cache size=117G
preencode exitcode=0
log=/root/wuqingman/logs/exp27_vjepa_cache_full.log
```

## 已完成 smoke

RTX 5090 已使用 V-JEPA 2.1 Large、真实 LIBERO batch 和真实 feature cache 完成单卡
forward/backward。该结果证明数据、cache、teacher target、loss 和梯度路径可运行，不是正式训练
效果或成功率证据。

## 正式训练启动

```bash
cd /root/wuqingman/Ego-WAM-exp27
export WORKSPACE_ROOT=/root/wuqingman
export VJEPA_CACHE_DIR=/root/wuqingman/datasets/vjepa_cache/libero_spatial
export ACCELERATE_PYTHON=/root/wuqingman/venvs/egowam-py311/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MAIN_PROCESS_PORT=29695
export JOB_NAME=27-0811_libero_spatial_vjepa_align_100k
export OUTPUT_DIR=$WORKSPACE_ROOT/RUN/$JOB_NAME
export swanlab_experiment=$JOB_NAME

nohup bash training/scripts/train_native_libero_spatial_vjepa_align_100k.sh \
  > $WORKSPACE_ROOT/logs/${JOB_NAME}.log 2>&1 &
```

已记录的 PID 为 `1780395`，但这是启动时快照，不能据此判断当前进程或训练完成状态。

## 验收与待补

正式训练必须记录 action/image/V-JEPA 三项 loss、有效 warmup 系数、LR、grad norm、吞吐、
OOM/restart、cache metadata 校验与 checkpoint。评测不加载 V-JEPA，必须与 Exp20 使用同 task、
trial、wait steps、normalization、reset protocol 和 flow seed。

待补：正式 H20 训练开始/结束时间、最终 loss、checkpoint SHA256、SwanLab、逐 checkpoint
LIBERO Spatial 500-episode 结果，以及无 alignment 的 Exp20 同协议 control。
