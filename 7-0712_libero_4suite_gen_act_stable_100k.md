# 7-0712_libero_4suite_gen_act_stable_100k

## 状态

```text
正式训练曾在 step 21,805 外部中断
最后 checkpoint=step_20000
已准备 tmux 恢复脚本
恢复后的最终状态需到对应作业确认
```

## 实验身份

```text
JOB_NAME=7-0712_libero_4suite_gen_act_stable_100k
branch=egowam-stable
code_commit=c20ef8aa91e4aaa30b8b4d846df88232042bd472
main_launcher=training/scripts/train_native_libero_all_gen_act.sh
suite_launcher=training/scripts/train_native_libero_4suite_gen_act.sh
```

## 科学问题

以实验 6 为严格基线，唯一 treatment 是加入训练时 Generation co-training：

```text
构建 Generation expert
从 SenseNova U1 初始化 Generation
Generation full train
image flow loss=1
Action flow loss=1
```

其他设置保持与实验 6 一致。

## 相对实验 6 的变量

```text
enable_gen_branch: false → true
load_gen_branch: false → true
disable_gen_forward: true → false
gen_train_mode: frozen → full
image_gen_loss_coef: 0 → 1
future image: absent → t+16
```

Action 保持：

```text
随机初始化
init_action_from_gen=false
```

## 模型

Smoke 实测：

```text
total/trainable params=2,696.87M
Understanding=997.43M
Generation=861.25M
Action=838.19M
```

初始化：

```text
Understanding=SenseNova 4096→1024
Generation=SenseNova Gen 4096→1024
Action=random
```

Checkpoint 加载：

```text
loaded tensors=1115
missing=473
unexpected=0
interpolated=772
```

## 数据

四套：

```text
libero_spatial
libero_object
libero_goal
libero_10
```

排除：

```text
libero_90
```

有效样本：

```text
62,672
```

每个 sample：

```text
instruction
current [agentview|wrist]
current 8D state
future [agentview|wrist] at t+16
16×7 action chunk
```

## Loss

```text
total_loss=image_gen_loss+action_gen_loss
LM CE=0
Value loss=0
```

## 优化

```text
GPUs=8
batch/GPU=32
global batch=256
steps=100,000
lr=1e-4
AdamW
bf16
FSDP FULL_SHARD
```

## Smoke

单卡：

```text
2/2 steps 完成
双 loss 正常
```

8 卡：

```text
10/10 steps 完成
batch/GPU=32
无 OOM
```

最后 smoke：

```text
total loss=2.8227
image loss=0.6305
action loss=2.1922
grad norm=8.34
```

回归：

```text
21 tests passed
```

## 正式训练中断

首次正式运行：

```text
运行到 step 21,805
最后完整 checkpoint=step_20000
```

日志中：

```text
无 Traceback
无 RuntimeError
无 OOM
无 NCCL error
磁盘充足
```

判断为 terminal/SIGHUP/外部终止，而非训练 Bug。

## 恢复

持久化脚本：

```text
/mnt/public/wuqingman/tools/resume_exp7.sh
```

建议 tmux：

```bash
cp /mnt/public/wuqingman/tools/resume_exp7.sh /root/resume_exp7.sh
chmod +x /root/resume_exp7.sh

tmux new-session -d -s exp7_resume \
  "bash -lc '
    set -o pipefail
    LOG=/root/wuqingman/logs/7-0712_libero_4suite_gen_act_stable_100k.log
    env RESUME_STEP=20000 MAIN_PROCESS_PORT=29695 \
      bash /root/resume_exp7.sh 2>&1 | tee -a \$LOG
  '"
```

## 完整记录

```text
/root/wuqingman/RUN/7-0712_libero_4suite_gen_act_stable_100k/
EXPERIMENT.md
```

## 后续

1. 确认恢复作业是否已经达到 100k。
2. 合并最终 checkpoint。
3. 使用与实验 6 完全一致的 replan=10、50 trials/task 做四套评测。
4. 重点比较 task 4/5/8，验证 Generation co-training 是否改善抓取和恢复。
