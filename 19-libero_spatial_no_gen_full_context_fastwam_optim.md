# 19-libero_spatial_no_gen_full_context_fastwam_optim

## Material Passport

```text
artifact_type=experiment_result
experiment_id=19
status=training_configuration_recorded_results_pending
verification_status=ANALYZED_FROM_PROVIDED_LAUNCH_MATERIAL
recorded_at=2026-08-13
```

## 状态

```text
训练配置已登记
运行时长：约 28h（附件记录）
训练完成、checkpoint、最终 loss：待补
正式 LIBERO 评测：待补
```

附件包含启动脚本与 SwanLab run 链接，但没有训练结束日志、实际退出码、checkpoint 清单、
最终 loss 或评测 `summary.csv`。因此本记录只确认设计和启动参数，不宣称训练已完成。

## 实验身份

```text
JOB_NAME=19-libero_spatial_no_gen_full_context_fastwam_optim
baseline=实验 18：18-libero_spatial_no_gen_fastwam_optim
code_commit=f98d358d09f566f263461e1923e8f3a58faf74b8
code_tarball_sha256=0f918ff246c864f233adfe8b99d7c08f424294991a6365b96745991278206478
SwanLab=http://10.170.31.44:8000/@wx1512724/EgoWAM-Native/runs/22n302jqxfggmwc4xu44y
```

附件中的 OBS/SwanLab 凭据不写入仓库。

## 科学问题

以实验 18 为基线，只改变 Understanding expert 的 context attention：

```text
Exp18: context_attention_mode=isolated
Exp19: context_attention_mode=full
```

这检验 Understanding 是否从 full context 中获益，以及该 attention topology 对 No-Gen
Spatial 闭环成功率的影响。

## 计划控制变量

```text
data=libero_spatial; views=image,wrist_image; No-Gen; Understanding + Action full train
frame_sampling_stride=1; action_horizon=32; action_exec_horizon=10; drop_incomplete_action_chunks=false; delta_action_dim_mask=true,true,true,true,true,true,false
hidden size=1024; robot_state_dim=8; state injection=und; action flow=direct v-prediction; shift=5; target_v=noise-action
normalization=min/max; eps=1e-8; GPUs=8; batch/GPU=32; global_batch=256; gradient_accumulation=1
total_steps=100000; lr=1e-4; optimizer=AdamW; betas=(0.9,0.95); lr_scheduler=cosine_with_min_lr; warmup_ratio=0.05; min_lr_ratio=0.01; weight_decay=0.01
gradient_checkpointing=false; dtype=bf16; SEED=42; num_workers=4; logging_steps=10; save_steps=5000; max_keep_ckpt=0
```

## 启动路径与协议风险

```text
dataset_obs=obs://yw-2030-gy/external/wx1512724/datasets/nvidia/LIBERO_LeRobot_v3/libero_spatial
model_obs=obs://yw-2030-gy/external/wx1512724/models/sensenova/SenseNova-U1-8B-MoT-SFT
local_output=/cache/wx1512724/RUN/19-libero_spatial_no_gen_full_context_fastwam_optim
uploaded_output=obs://yw-2030-gy/external/wx1512724/training_runs/19-libero_spatial_no_gen_full_context_fastwam_optim/latest_run/
training_log=/cache/wx1512724/logs/ads_exp19_train.log
```

启动脚本设置 `SCAN_DIR` 为 Spatial 数据目录，但实际调用的是：

```text
bash scripts/train_native_libero_all_no_gen.sh
```

这与 Exp18 的 Spatial 专用 launcher 不同。必须从实际训练日志或 launcher 内容确认该脚本
是否读取 `SCAN_DIR` 并只训练 Spatial；在确认前，Exp19 不能被视为严格的单变量 Exp18 复现。

## 正式评测协议（待训练完成）

应与 Exp18/Exp22 明确对齐：

```text
suite=libero_spatial; MuJoCo=3.3.2; tasks=10; trials/task=50
action_horizon=32; action_exec_horizon/replan=10; num_inference_steps=10
num_wait_steps=5（若沿用 Exp22）; disable_gen_branch=true
normalization=min/max; stats=<Exp19 run>/dataset_stats.json
context_attention_mode=full; action-flow seed=必须记录
```

Exp22 的 60k 结果为 `428/500=85.6%`，Exp10 的 step 90000 为 `454/500=90.8%`，但两者
评测协议存在 wait steps 差异，只能作为参考，不能替代同协议 Exp19 对照。

## 当前结论边界与待补

目前只能确认 Exp19 的预设 treatment 是 `isolated -> full`，不能确认实际训练是否完成，
也不能确认 launcher 是否真的只使用 Spatial 数据。

待补：实际训练日志与退出码；有效样本数；最终 LR/loss/grad norm；checkpoint 列表和 SHA256；
正式 50-trial/task 评测 CSV；与 Exp18 相同 init state 和 action-flow seed 的配对评测。
