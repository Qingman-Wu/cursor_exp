# Ego-WAM 正式实验索引

> 目录规则：每个正式实验一个 Markdown，文件名严格等于 `JOB_NAME`。
> Smoke、临时 debug、单元测试和短链路检查不单独建实验文档。

## 已登记实验

### 已完成

1. [`5-0709_libero_spatial_no_gen_stable_50k`](./5-0709_libero_spatial_no_gen_stable_50k.md)
   - LIBERO Spatial，No-Gen，50k
   - 最终训练 loss：0.0065
   - Spatial 500-episode 最佳结果：88.0%（replan=10）

2. [`6-0711_libero_4suite_no_gen_stable_100k`](./6-0711_libero_4suite_no_gen_stable_100k.md)
   - 标准四套 LIBERO，No-Gen，100k
   - 最终训练 loss：0.0150
   - Step85000 正式评测：71.8%（1436/2000 episodes）

3. [`8-eval_exp6_step85000`](./8-eval_exp6_step85000.md)
   - 评测实验 6 的 step 85000 checkpoint
   - 四套 LIBERO，50 trials/task，共 2000 episodes
   - Overall 71.8%；Spatial 74.0%；Object 90.6%；Goal 68.8%；LIBERO-10 53.8%

4. [`10-0713_libero_spatial_no_gen_fastwam_align_100k`](./10-0713_libero_spatial_no_gen_fastwam_align_100k.md)
   - LIBERO Spatial，No-Gen，FastWAM-aligned 数据/控制设置，100k
   - 最终训练 loss：0.0027
   - 正式评测待完成

### 中止或被替代

5. [`5-0709_libero_spatial_no_gen_stable_10k`](./5-0709_libero_spatial_no_gen_stable_10k.md)
   - 原计划 10k，实际只运行到 step 724
   - 被正式 50k 实验替代

6. [`6-0710_libero_all_no_gen_stable_100k`](./6-0710_libero_all_no_gen_stable_100k.md)
   - 包含 `libero_90` 的五套训练
   - 运行到 step 11,364 后主动停止
   - 原因：`libero_90` 占比过高，不符合标准 VLA 四套协议

### 跨作业运行 / 状态待确认

7. [`7-0712_libero_4suite_gen_act_stable_100k`](./7-0712_libero_4suite_gen_act_stable_100k.md)
   - 标准四套 LIBERO，Generation + Action，100k
   - 首次运行到 step 21,805 后外部中断，随后从 step 20,000 恢复
   - 已完成 100,000 steps

### 运行中

8. [`12-0716_eval_exp7`](./12-0716_eval_exp7.md)
   - 实验 7 Gen+Act 的 step 80000、step 90000 与 step 100000 正式四套 LIBERO 评测
   - 三个 checkpoint 当前同时共享 8×H20
   - 50 trials/task，replan=10，共 6000 episodes
   - 启动前三个 checkpoint smoke 均为 1/1 success

9. [`14-0723_libero_spatial_no_gen_fastwam_optim_align_100k`](./14-0723_libero_spatial_no_gen_fastwam_optim_align_100k.md)
   - 以实验 10 为基线，对齐 FastWAM 的 LR schedule、weight decay 和 checkpointing
   - batch/GPU 保持 32，global batch 保持 256
   - 8 卡 20-step smoke 已通过，正式训练待启动

## 不收录为正式实验

以下仅为 smoke/debug，不单独建文档：

```text
5-0709_libero_spatial_no_gen_stable_smoke50
7-exp7_gen_act_single_gpu_smoke
7-exp7_gen_act_fsdp_bs32_smoke
7-exp7_gen_act_fsdp_memory_smoke
```

## 统一要求

每个实验文档至少记录：

```text
JOB_NAME
目标与对照
代码 commit
数据集和有效样本数
模型分支与初始化
Normalization
Action horizon / stride / replan
Batch / GPU / global batch
Optimizer / LR / steps
训练状态和最终 loss
Checkpoint
SwanLab
正式评测结果
失败原因或后续待办
```

## 相关总交接

```text
/mnt/public/wuqingman/cursor_chat/
2026-07-10_to_2026-07-13_libero_training_evaluation_handoff.md
```

## LIBERO 正确评测手册

```text
/mnt/public/wuqingman/libero_eval/LIBERO_EVALUATION.md
```

## FastWAM诊断实验复现手册

```text
/mnt/public/wuqingman/cursor_exp/FastWAM_DIAGNOSTICS_REPRODUCTION.md
```

该文档包含本次全部诊断实验的方法、控制变量、指标、执行顺序、完整Python代码和
Ego-WAM核心诊断diff，供其他机器上的AI适配官方FastWAM `.pt`权重。
