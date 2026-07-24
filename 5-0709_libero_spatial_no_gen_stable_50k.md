# 5-0709_libero_spatial_no_gen_stable_50k

## 状态

```text
训练完成
正式 Spatial 评测完成
```

## 实验身份

```text
JOB_NAME=5-0709_libero_spatial_no_gen_stable_50k
branch=egowam-stable
code_commit=515464a7055ab26fc66f975425ea847f5ffb93d2
SwanLab=https://swanlab.cn/@woovine/EgoWAM-Native/runs/0ajnf35m
```

## 实验目标

在 LIBERO Spatial 上验证无 Generation expert 的 action-only VLA 能否从随机 Action
初始化收敛，并完成闭环操作。

## 数据

```text
dataset=nvidia/LIBERO_LeRobot_v3/libero_spatial
format=LeRobot v3
effective samples=11,792
frame_sampling_stride=4
action_horizon=16
drop_incomplete_action_chunks=true
```

图像：

```text
agentview 256×256
wrist 256×256
horizontal concat [image|wrist]
final visual size=512×256
patch grid=16×32
IMG_CONTEXT tokens=128
```

## 模型

```text
model=EgoWAMNativeModel
total/trainable params=1,835.62M
Understanding=997.43M, full train
Generation=not built
Action=838.19M, full train, random init
LM CE=disabled
Value=disabled
```

Hidden width：

```text
lang=1024
gen=1024（分支未构建）
act=1024
```

Robot state：

```text
dim=8
history=1
tokens=1
injection=Understanding
encoder=Linear(8,1024)
```

## Normalization

```text
mode=z-score
normalize_action_state=true
eps=1e-6
stats=libero_spatial/meta/stats.json
```

## Action Flow Matching

```text
direct v-prediction
z=(1-sigma)*action+sigma*noise
target=noise-action
shift=5
num_train_timesteps=1000
```

## 优化

```text
GPUs=8
batch/GPU=32
grad_accum=1
global batch=256
steps=50,000
lr=1e-4
optimizer=AdamW
betas=(0.9,0.95)
weight_decay=0
precision=bf16
FSDP=FULL_SHARD
```

## 训练结果

```text
50000/50000 完成
训练耗时≈18h27m
final action loss=0.0065
无 OOM / Traceback / RuntimeError
```

Checkpoint：

```text
/root/wuqingman/RUN/5-0709_libero_spatial_no_gen_stable_50k/
step_50000
step_50000_hf/model.safetensors
```

## 开环验证

```text
训练 DataLoader pack action MSE≈0.00019
评测 Policy pack action MSE≈0.00022
```

证明模型、权重加载、图像、state、prompt 和 normalization 正确。

## 闭环评测修复

最初错误评测接近 0%，后来修复：

```text
gripper 0/1 → env -1/+1
MuJoCo visual mesh
state/action normalization
图像旋转和双视角顺序
no-gen model structure
FSDP checkpoint merge
```

评测 codebase：

```text
egowam-stable@c20ef8a
MuJoCo 3.3.2
robosuite 1.4.1
20 flow inference steps
50 trials/task
```

## Replan 正式评测

### Replan=4

```text
412/500 = 82.4%
```

### Replan=10

```text
440/500 = 88.0%
```

### Replan=16

```text
428/500 = 85.6%
```

最终推荐：

```text
action_horizon=16
action_exec_horizon/replan=10
num_inference_steps=20
```

## 失败规律

主要失败集中在 task 4/5/8：

- 精细抓取失败；
- 漏抓后空手继续执行 place phase；
- Gripper 在多个 replan 间抖动；
- 同一 init state 因 flow noise 可能一次成功、一次失败。

视频和分析：

```text
/root/wuqingman/RUN/5-0709_libero_spatial_no_gen_stable_50k/
pattern_videos_step50000/README.md
```

## 结论

No-Gen Action-only 模型能够在 Spatial 上达到 88.0%，证明 Ego-WAM action 分支和评测
链路有效。当前主要瓶颈是精细抓取、gripper 稳定性和失败恢复能力。
