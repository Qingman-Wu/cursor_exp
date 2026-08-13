# 8-eval_exp6_step85000

## Material Passport

```text
artifact_type=experiment_result
experiment_id=8
evaluation_target=6-0711_libero_4suite_no_gen_stable_100k
checkpoint=step_85000
status=completed
verification_status=ANALYZED_FROM_REPORTED_SUMMARY
recorded_at=2026-08-13
```

## 状态

```text
已完成
四套 LIBERO × 10 tasks × 50 trials = 2000 episodes
overall=1436/2000=71.8%
```

本实验是实验 6 的正式闭环评测，不是新的训练实验。原始运行日期未在现有材料中提供，
因此文件名不补写未经验证的日期。

## 实验身份

```text
EXPERIMENT_ID=8
evaluation_target=6-0711_libero_4suite_no_gen_stable_100k
training_code_commit=8159aa38cb2f3159b7a13a90c5d033db367c64d3
evaluation_code_commit=未提供
checkpoint=/root/wuqingman/RUN/6-0711_libero_4suite_no_gen_stable_100k/step_85000_hf/model.safetensors
launcher=/root/run_step85000_full_eval.sh
log=/root/wuqingman/logs/eval_step85000_full.log
```

## 评测协议

```text
suites=libero_spatial,libero_object,libero_goal,libero_10
tasks/suite=10
trials/task=50
episodes/suite=500
total episodes=2000
GPUs=0,1,2,3,4,5,6,7
MuJoCo=3.3.2（由 egowam-libero-mj332 环境名记录）
render_backend=EGL
EGL_vendor=/root/wuqingman/egl_vendor/10_nvidia.json
stats=/root/wuqingman/eval_stats/<suite>/stats.json
```

训练实验 6 文档记录评测使用 `replan=10`。本次提供的总控脚本通过
`run_libero_parallel.sh` 间接调用 evaluator；该下游脚本内容尚未归档，因此这里不额外推断
inference steps、wait steps 等未在材料中直接出现的参数。

## 汇总结果

| Suite | Successes | Episodes | Success rate |
|---|---:|---:|---:|
| libero_spatial | 370 | 500 | 74.0% |
| libero_object | 453 | 500 | 90.6% |
| libero_goal | 344 | 500 | 68.8% |
| libero_10 | 269 | 500 | 53.8% |
| **Overall** | **1436** | **2000** | **71.8%** |

逐任务原始结果保存在：

```text
results/8-eval_exp6_step85000/summary.csv
```

## Spatial 逐任务结果

| Task | Successes | Trials | Success rate |
|---:|---:|---:|---:|
| 0 | 35 | 50 | 70% |
| 1 | 42 | 50 | 84% |
| 2 | 47 | 50 | 94% |
| 3 | 50 | 50 | 100% |
| 4 | 30 | 50 | 60% |
| 5 | 33 | 50 | 66% |
| 6 | 49 | 50 | 98% |
| 7 | 15 | 50 | 30% |
| 8 | 34 | 50 | 68% |
| 9 | 35 | 50 | 70% |

## 结论边界

这份结果证明同一 Ego-WAM No-Gen 基线在现有 LIBERO 闭环链路上并非只能达到约 0.1：
实验 6 的 step 85000 在 Spatial 达到 74.0%，四套总体达到 71.8%。它不能证明后续实验的
训练或评测链路一定正确，但为排查实验 20 以后 Spatial 低成功率提供了一个已知可工作的
checkpoint、环境和评测协议锚点。

实验 6 的 Spatial 并未达到 98% 的 suite 平均成功率；98% 只出现在 Spatial task 6，
不能将单任务结果当作 suite 平均结果。

## 可复现脚本

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[ERROR] line=$LINENO command=$BASH_COMMAND exit=$?" >&2' ERR

ROOT=/root/wuqingman
JOB=6-0711_libero_4suite_no_gen_stable_100k
STEP=85000
TRIALS=${TRIALS:-50}
GPUS=0,1,2,3,4,5,6,7

export RUN_DIR=$ROOT/RUN/$JOB
export TRAIN=$ROOT/Ego-WAM-egowam-stable/training
export LIBBIN=$ROOT/venvs/egowam-libero-mj332/bin
export BASE_MODEL=$ROOT/models/SenseNova-U1-8B-MoT-SFT
export STATS_ROOT=$ROOT/eval_stats
export EVAL_ROOT=$RUN_DIR/eval_mj332_step${STEP}_${TRIALS}trials

export __EGL_VENDOR_LIBRARY_FILENAMES=$ROOT/egl_vendor/10_nvidia.json
export LIBERO_RENDER_BACKEND=egl
export MUJOCO_EGL_DEVICE_ID=0

test -f "$__EGL_VENDOR_LIBRARY_FILENAMES"
test -f "$RUN_DIR/step_${STEP}_hf/model.safetensors"
test -f "$BASE_MODEL/model.safetensors.index.json"
test -x "$LIBBIN/python"
test -x "$ROOT/run_libero_parallel.sh"

mkdir -p "$EVAL_ROOT"

for suite in libero_spatial libero_object libero_goal libero_10; do
  echo
  echo "=================================================="
  echo "Suite: $suite"
  echo "Tasks: 10"
  echo "Trials per task: $TRIALS"
  echo "=================================================="

  export ACTION_STATE_STATS_PATH=$STATS_ROOT/$suite/stats.json
  export OUT_DIR=$EVAL_ROOT/$suite
  export LOG_DIR=$EVAL_ROOT/logs_$suite

  test -f "$ACTION_STATE_STATS_PATH"
  test -f "$ROOT/eval_tasks_libero/$suite.txt"

  rm -rf "$OUT_DIR" "$LOG_DIR"
  mkdir -p "$OUT_DIR" "$LOG_DIR"

  bash "$ROOT/run_libero_parallel.sh" \
    "$suite" \
    "$STEP" \
    "$TRIALS" \
    "$GPUS"

  "$LIBBIN/python" - "$OUT_DIR" "$TRIALS" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
trials = int(sys.argv[2])

files = sorted(
    path for path in out.glob("libero_*.json")
    if path.name != "summary.json"
)

assert len(files) == 10, (
    f"{out}: expected 10 task JSON files, got {len(files)}"
)

for path in files:
    payload = json.loads(path.read_text())
    assert payload["total"] == trials, (
        f"{path}: expected {trials} trials, got {payload['total']}"
    )

print(f"[OK] {out.name}: 10 tasks x {trials} trials")
PY
done

cd "$TRAIN"

PATH="$LIBBIN:$PATH" \
PYTHONPATH="$TRAIN" \
"$LIBBIN/python" - "$EVAL_ROOT" <<'PY'
import sys
from pathlib import Path

from egowam_native.eval.common import load_task_results, write_summary

root = Path(sys.argv[1])
suites = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
]

results = []
for suite in suites:
    suite_results = load_task_results(root / suite)
    assert len(suite_results) == 10, (
        f"{suite}: expected 10 task results, got {len(suite_results)}"
    )
    results.extend(suite_results)

assert len(results) == 40
summary = write_summary(results, root)
print("FINAL OVERALL:", summary["overall"])
print("summary.csv:", root / "summary.csv")
print("summary.json:", root / "summary.json")
PY

cat "$EVAL_ROOT/summary.csv"
```

后台启动命令：

```bash
nohup bash /root/run_step85000_full_eval.sh \
  > /root/wuqingman/logs/eval_step85000_full.log \
  2>&1 &

echo $! > /root/wuqingman/RUN/eval_step85000.pid
```

## 待补材料

1. 本次评测实际开始与完成时间。
2. 评测时 Ego-WAM 的代码 commit。
3. `/root/run_libero_parallel.sh` 的版本或内容。
4. `summary.csv` 原文件的 checksum；当前 CSV 根据用户提供的结果截图转录。
