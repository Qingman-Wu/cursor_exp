# FastWAM 权重诊断实验完整复现手册

> 文档用途：把本机针对 Ego-WAM Experiment 7 完成的全部 LIBERO 诊断方法、代码和
> 结果判读方式交给另一台机器上的 AI，由它适配官方 FastWAM `.pt` 权重。
>
> 重要：本文附带的是 **Ego-WAM 参考实现**，不是可直接运行的 FastWAM 脚本。
> FastWAM AI 需要替换模型加载、observation/action adapter、dataset loader 和 module
> hook 名称；实验设计、控制变量、统计方式和产物格式应保持不变。

---

## 1. 诊断目标

不要只回答“FastWAM 成功率是多少”，而要区分以下原因：

1. 正式评测统计或 adapter 错误；
2. checkpoint 未正确加载、normalization/stats 错误；
3. action flow/diffusion 噪声导致不可复现；
4. 模型在训练数据开环上就拟合不好；
5. 开环好但闭环发生 compounding error；
6. gripper 时序、action chunk、replan 或 ensemble 导致失败；
7. 模型过度依赖 proprioceptive state，未使用视觉判断物体/抓取状态；
8. checkpoint 间发生任务干扰或遗忘；
9. Video/Generation loss 与 Action loss 存在梯度干扰；
10. 图像方向、相机顺序或实时渲染与训练数据不一致。

最终产物必须包含：

```text
逐 suite / task 成功率
跨 checkpoint 变化排名
固定 task/init/noise 的配对视频
逐步 observation/state/action/flow 轨迹
开环 GT 误差与 gripper accuracy
图像/state/语言消融
模型内部 activation 与 cosine
多 loss 梯度 cosine
replan/hysteresis/ensemble/image-transform 对照
证据分级的根因报告
```

---

## 2. FastWAM AI 首先必须实现的最小接口

所有参考脚本围绕下面这个抽象接口。FastWAM 侧应先包装官方模型：

```python
class FastWAMDiagnosticPolicy:
    def predict_action_chunk(self, observation: dict) -> np.ndarray:
        # 返回 dataset-space action chunk，shape=[H, D]。
        ...

    def reset_flow_seed(self, seed: int) -> None:
        # 重置 action diffusion/flow 初始噪声及 replan counter。
        ...

    def get_last_diagnostics(self) -> dict:
        # 返回最近一次推理的 normalized/denormalized action 和采样轨迹。
        ...
```

统一 observation schema（FastWAM AI 可在内部转成官方格式）：

```python
observation = {
    "instruction": str,
    "image": np.ndarray,        # agentview RGB
    "wrist_image": np.ndarray,  # wrist RGB
    "robot_state": np.ndarray,  # FastWAM训练时的proprio schema
}
```

`get_last_diagnostics()` 推荐返回：

```python
{
    "sample_seed": int,
    "initial_noise": np.ndarray,
    "normalized_action": np.ndarray,
    "denormalized_action": np.ndarray,
    "flow_states": np.ndarray,      # [num_steps+1, H, D]
    "velocity_norms": np.ndarray,   # [num_steps]
    "timesteps": np.ndarray,
    "deltas": np.ndarray,
}
```

FastWAM 适配时必须确认：

```text
checkpoint格式=.pt
dataset_stats与该训练run完全对应
state/action normalization与训练一致
agent/wrist顺序与训练一致
图像resize/crop/normalize与训练一致
gripper dataset convention与LIBERO env convention的转换正确
action horizon / n_action_steps / replan与被测协议一致
```

---

## 3. 推荐目录结构

```text
FASTWAM_ROOT/
├── checkpoints/
│   ├── step_80000.pt
│   ├── step_90000.pt
│   ├── step_100000.pt
│   └── dataset_stats.json
├── diagnostics/
│   ├── tools/                    # 本文附带脚本适配版
│   ├── task_profile/
│   ├── paired_rollouts/
│   ├── openloop/
│   ├── internal/
│   ├── gradients/
│   ├── interventions/
│   └── report/
└── official_FastWAM_repo/
```

每次运行必须保存：

```text
FastWAM git commit
checkpoint路径和SHA256
dataset stats路径和SHA256
Python/Torch/CUDA/MuJoCo版本
task、trial、init state id
env seed和action flow seed
action horizon、replan、inference steps
相机尺寸、顺序和图像变换
完整执行命令
```

---

## 4. 实验执行顺序

必须按顺序执行。前面的契约不正确时，后面的模型内部结论没有意义。

### 实验 A：正式结果三层校验

目的：

```text
确认summary、task JSON、逐trial log完全一致；
确认每个task都有固定数量trial；
防止部分任务失败退出后仍生成错误overall。
```

输入：

```text
summary.csv
每task一个JSON
每task一个逐trial log
```

验收：

```text
40/40 tasks verified
每task trial IDs完整（如0..49）
summary successes = JSON episode successes = log success=True计数
overall = 40个task求和
mismatches=[]
```

参考代码：`verify_eval_results.py`。

### 实验 B：逐任务失败率画像

目的：

```text
映射task ID到自然语言；
找各checkpoint最低成功率task；
找checkpoint间提升/退化最大的task；
避免overall掩盖任务遗忘。
```

输出：

```text
task_profile.json
task_profile.csv
task_profile.md
```

参考代码：`analyze_libero_results.py`。

### 实验 C：选择配对trial

对同一task的同一个trial比较多个checkpoint，优先选：

```text
全部失败；
早期失败、后期成功；
早期成功、后期失败；
全部成功对照。
```

参考代码：`select_paired_trials.py`。

### 实验 D：固定action flow/diffusion噪声

正式评测中的env seed不一定固定模型采样噪声。必须让：

```text
task相同
trial相同
replan index相同
→ 所有checkpoint使用相同initial action noise
```

推荐seed公式：

```python
sample_seed = base_flow_seed + task_id * 1_000 + trial_id * 100_000 + replan_index
```

不要依赖全局`torch.manual_seed`，应为每个device创建独立`torch.Generator`。

参考实现见后文核心diff中的`policy.py`。

### 实验 E：配对闭环视频与完整轨迹

固定：

```text
task
trial/init state
flow seed sequence
normalization
replan
inference steps
```

每个replan保存：

```text
agentview/wrist原图和模型输入图
raw/normalized state
raw/normalized/denormalized action chunk
最终送入env的action
gripper阈值前后值
initial noise
完整flow trajectory
每步velocity/prediction norm
```

整条episode保存：

```text
MP4
executed_actions.npy
diagnostics.npz
success
steps
```

统计：

```text
首次close时间
gripper transitions
open/close占比
平移/旋转动作幅度
state path length
replan数
是否跑满step limit
```

参考：`libero_vis.py`核心diff、`summarize_rollout_diagnostics.py`。

### 实验 F：开环Teacher-Forcing

从FastWAM训练数据取真实：

```text
image + wrist image + state + instruction + GT action chunk
```

用评测policy pack和sampler预测，计算：

```text
overall action MAE/MSE
per-dimension MAE
translation/rotation误差
gripper accuracy/F1
chunk前段/后段误差
多flow seed prediction std
```

必须只使用当前timestep的图像，不能误用future target image。

参考：`openloop_condition_diagnostics.py`。

### 实验 G：图像/state/语言消融

固定同一sample和initial noise，仅替换：

```text
baseline
zero agentview
zero wrist
zero all images
zero state
wrong/empty instruction
```

指标：

```text
mean |action_condition - action_baseline|
第一步动作变化
MAE vs GT
gripper变化
```

判读：

```text
zero state影响远大于zero image → state shortcut
zero image影响大 → 模型真实使用视觉
wrong instruction几乎无影响 → 语言条件弱或任务主要靠视觉区分
```

### 实验 H：模型内部Activation Hook

FastWAM需要自行映射模块名称，建议hook：

```text
vision patch/projector
proprio/state encoder
text encoder输出
Action DiT第0/中间/最后层
action head
time/noise embedding
```

每层保存：

```text
shape
mean/std/norm
token pooled vector
不同输入条件下cosine
不同checkpoint同task cosine
同checkpoint低分/高分task cosine
```

参考：`internal_activation_diagnostics.py`、`compare_activation_cosine.py`。

### 实验 I：Gen/Video loss与Action loss梯度关系

在同一个真实训练batch上分别反向：

```python
video_grads = torch.autograd.grad(video_loss, selected_params, retain_graph=True)
action_grads = torch.autograd.grad(action_loss, selected_params)
```

参数组：

```text
observation vision encoder
shared multimodal backbone
state encoder
Action expert
Video/Generation expert
```

统计每组：

```text
video gradient norm
action gradient norm
dot product
cosine similarity
```

必须同时测低分与高分task。单个batch的负cosine不能直接证明长期训练有害；
只有当低分task/大量batch系统性更冲突时才能归因。

参考：`gradient_conflict_diagnostics.py`。FastWAM需要把`image_gen_loss`替换为官方
video/world loss字段。

### 实验 J：控制与adapter消融

固定task/trial/noise，对照：

```text
replan / n_action_steps：训练匹配值 vs 正式值
action ensemble：on/off
gripper hysteresis：on/off
image transform：none/vertical/rotate180
单checkpoint运行 vs 多checkpoint并发
```

Hysteresis参考：

```text
close_threshold=0.35
open_threshold=0.65
min_hold_steps=10
```

不要只看单trial成功，还要记录gripper transitions和完成步数。

### 实验 K：No-Video/No-Generation严格对照

如果有FastWAM action-only checkpoint，必须保证：

```text
相同训练数据
相同训练steps
相同batch/optimizer
相同Action初始化
相同eval task/init/noise
```

否则“有Video vs 无Video”会和数据量、suite或checkpoint差异混淆。

---

## 5. 结果判读矩阵

```text
开环差、闭环差：
  优先查模型拟合、normalization、checkpoint、sampler。

开环好、闭环差：
  优先查covariate shift、action chunk、gripper和失败恢复。

低分task zero-state影响大、zero-image影响小：
  state shortcut；模型未使用物体/抓取视觉状态。

gripper transitions多，hysteresis后仍失败：
  抖动是症状，真正问题是抓取/恢复策略。

ensemble只在step limit附近成功：
  边缘轨迹被平滑救回，不代表稳定能力。

同task跨checkpoint差异大：
  task interference、遗忘或checkpoint specialization。

低分task梯度冲突不比高分task强：
  不能把失败简单归因于Video/Action gradient conflict。

改变图像方向后行为更差：
  原方向更可能正确；仍需用同init dataset-vs-sim图做最终确认。
```

---

## 6. 本机Ego-WAM诊断得到的参考现象

这些数字只用于帮助FastWAM AI理解输出，不应当作为FastWAM预期结果：

```text
task4同task/trial/noise：
80k fail, 410 steps, gripper transitions=22
90k fail, 410 steps, gripper transitions=10
100k success, 127 steps, gripper transitions=1
No-Gen success, 157 steps, gripper transitions=5

90k低分task4：
zero-images action change=0.0276
zero-state action change=0.1485

90k高分task2：
zero-images action change=0.2107
zero-state action change=0.0461
```

---

## 7. FastWAM适配检查清单

另一台AI完成适配后，逐项确认：

```text
[ ] .pt checkpoint加载后打印missing/unexpected keys
[ ] dataset_stats与训练run匹配并记录SHA256
[ ] observation image/state/task字段与训练一致
[ ] action输出已反归一化
[ ] gripper映射正确
[ ] flow/diffusion初始噪声可按replan固定
[ ] open-loop同sample同seed可复现
[ ] paired rollout保存视频、动作和诊断NPZ
[ ] hooks覆盖vision/state/action backbone/head
[ ] gradient实验分别反向video loss与action loss
[ ] 所有实验保存完整命令和环境版本
```

---

## 8. 参考执行命令模板

逐task画像：

```bash
python tools/analyze_libero_results.py \
  --run "80k=/path/to/80k/results" \
  --run "90k=/path/to/90k/results" \
  --run "100k=/path/to/100k/results" \
  --output-dir /path/to/diagnostics/task_profile
```

三层结果校验：

```bash
python tools/verify_eval_results.py \
  --label 80k \
  --results /path/to/80k/results \
  --logs /path/to/80k/task_logs \
  --output /path/to/diagnostics/verification/80k.json
```

开环与条件消融（FastWAM适配后参数名可不同）：

```bash
python tools/openloop_condition_diagnostics.py \
  --root /path/to/LIBERO_dataset/libero_spatial \
  --suite libero_spatial \
  --task-id 4 \
  --checkpoint /path/to/fastwam_step90000.pt \
  --checkpoint-label 90k \
  --stats /path/to/dataset_stats.json \
  --gpu-id 0 \
  --flow-seeds 101,202,303 \
  --output /path/to/diagnostics/openloop/90k_spatial4.json
```

内部activation：

```bash
python tools/internal_activation_diagnostics.py \
  --root /path/to/LIBERO_dataset/libero_spatial \
  --suite libero_spatial \
  --task-id 4 \
  --checkpoint /path/to/fastwam_step90000.pt \
  --checkpoint-label 90k \
  --stats /path/to/dataset_stats.json \
  --gpu-id 0 \
  --output /path/to/diagnostics/internal/90k_spatial4.json
```

---

## 9. 附录说明

后面附带：

1. 所有独立诊断Python脚本全文；
2. 正式多卡评测启动脚本参考；
3. Ego-WAM核心诊断改动的完整git diff；
4. contract tests，供FastWAM AI重写后建立自己的测试。



# 附录A：独立诊断脚本全文


## `analyze_libero_results.py`

来源：`/root/wuqingman/analysis_exp7/analyze_libero_results.py`

```python
"""Generate a task-level LIBERO failure profile from rollout JSON files."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def _task_map(libero_src: Path) -> dict[str, list[str]]:
    sys.path.insert(0, str(libero_src))
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map

    return libero_task_map


def _load_runs(specs: list[str], task_map: dict[str, list[str]]) -> list[dict]:
    rows = []
    for spec in specs:
        label, raw_path = spec.split("=", 1)
        root = Path(raw_path)
        for path in sorted(root.glob("*.json")):
            if path.name == "summary.json":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            task_name = payload.get("task_name")
            if not task_name or ":" not in task_name:
                continue
            suite, task_id_raw = task_name.split(":", 1)
            task_id = int(task_id_raw)
            successes = int(payload["successes"])
            total = int(payload["total"])
            rows.append(
                {
                    "checkpoint": label,
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": task_name,
                    "description": task_map[suite][task_id].replace("_", " "),
                    "successes": successes,
                    "total": total,
                    "success_rate": successes / total if total else 0.0,
                    "result_path": str(path),
                }
            )
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict], path: Path) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["checkpoint"]].append(row)

    lines = ["# Experiment 7 LIBERO task failure profile", ""]
    for checkpoint in sorted(grouped):
        items = sorted(grouped[checkpoint], key=lambda row: (row["suite"], row["task_id"]))
        successes = sum(row["successes"] for row in items)
        total = sum(row["total"] for row in items)
        lines.extend(
            [
                f"## {checkpoint}",
                "",
                f"Completed tasks: {len(items)}/40; completed episodes: {successes}/{total} "
                f"({100 * successes / total:.1f}%)" if total else "No completed tasks.",
                "",
            ]
        )
        by_suite: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            by_suite[item["suite"]].append(item)
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            suite_rows = by_suite.get(suite, [])
            if not suite_rows:
                continue
            suite_successes = sum(row["successes"] for row in suite_rows)
            suite_total = sum(row["total"] for row in suite_rows)
            lines.extend(
                [
                    f"### {suite}",
                    "",
                    f"Completed: {len(suite_rows)}/10; {suite_successes}/{suite_total} "
                    f"({100 * suite_successes / suite_total:.1f}%)",
                    "",
                    "| Task | Success | Description |",
                    "|---:|---:|---|",
                ]
            )
            for row in sorted(suite_rows, key=lambda value: value["success_rate"]):
                lines.append(
                    f"| {row['task_id']} | {row['successes']}/{row['total']} "
                    f"({100 * row['success_rate']:.1f}%) | {row['description']} |"
                )
            lines.append("")

    by_task: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_task[(row["suite"], row["task_id"])].append(row)
    comparable = [items for items in by_task.values() if len(items) >= 2]
    lines.extend(
        [
            "## Cross-checkpoint variation",
            "",
            "| Suite/task | Min | Max | Range | Description |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for items in sorted(
        comparable,
        key=lambda values: max(row["success_rate"] for row in values)
        - min(row["success_rate"] for row in values),
        reverse=True,
    ):
        lo = min(items, key=lambda row: row["success_rate"])
        hi = max(items, key=lambda row: row["success_rate"])
        lines.append(
            f"| {lo['task_name']} | {lo['checkpoint']} {100 * lo['success_rate']:.1f}% | "
            f"{hi['checkpoint']} {100 * hi['success_rate']:.1f}% | "
            f"{100 * (hi['success_rate'] - lo['success_rate']):.1f} pp | {lo['description']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="LABEL=/path/to/results")
    parser.add_argument("--libero-src", default="/root/wuqingman/LIBERO-src")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = _load_runs(args.run, _task_map(Path(args.libero_src)))
    rows.sort(key=lambda row: (row["checkpoint"], row["suite"], row["task_id"]))
    (output / "task_profile.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(rows, output / "task_profile.csv")
    _write_markdown(rows, output / "task_profile.md")
    print(f"wrote {len(rows)} completed task rows to {output}")


if __name__ == "__main__":
    main()

```


## `verify_eval_results.py`

来源：`/root/wuqingman/analysis_exp7/verify_eval_results.py`

```python
"""Cross-check LIBERO summary CSV, task JSON, and per-trial logs."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


TRIAL_RE = re.compile(r"\[libero\]\s+(\S+)\s+trial=(\d+)\s+success=(True|False)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results)
    logs_dir = Path(args.logs)
    with (results_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    task_summary = {
        row["task_name"]: (int(row["successes"]), int(row["total"]))
        for row in summary_rows
        if row["task_name"] != "__overall__"
    }
    overall_row = next(row for row in summary_rows if row["task_name"] == "__overall__")
    mismatches = []
    verified = []
    suite_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for task_name, (summary_successes, summary_total) in sorted(task_summary.items()):
        safe = task_name.replace(":", "_")
        json_path = results_dir / f"{safe}.json"
        log_path = logs_dir / f"{safe}.log"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        episodes = payload["episodes"]
        json_successes = sum(bool(item["success"]) for item in episodes)
        json_total = len(episodes)
        log_trials = [
            (int(match.group(2)), match.group(3) == "True")
            for match in TRIAL_RE.finditer(log_path.read_text(encoding="utf-8", errors="replace"))
            if match.group(1) == task_name
        ]
        log_successes = sum(success for _, success in log_trials)
        log_total = len(log_trials)
        log_ids = sorted(trial_id for trial_id, _ in log_trials)
        expected_ids = list(range(summary_total))
        values = {
            "summary": [summary_successes, summary_total],
            "json": [json_successes, json_total],
            "log": [log_successes, log_total],
        }
        if len(set(tuple(value) for value in values.values())) != 1 or log_ids != expected_ids:
            mismatches.append(
                {
                    "task": task_name,
                    "counts": values,
                    "log_trial_ids_ok": log_ids == expected_ids,
                }
            )
        else:
            verified.append(task_name)
        suite = task_name.split(":", 1)[0]
        suite_totals[suite][0] += summary_successes
        suite_totals[suite][1] += summary_total

    computed_successes = sum(value[0] for value in task_summary.values())
    computed_total = sum(value[1] for value in task_summary.values())
    overall_csv = [int(overall_row["successes"]), int(overall_row["total"])]
    overall_computed = [computed_successes, computed_total]
    if overall_csv != overall_computed:
        mismatches.append({"overall_csv": overall_csv, "overall_computed": overall_computed})

    report = {
        "checkpoint": args.label,
        "verified_tasks": len(verified),
        "expected_tasks": 40,
        "mismatches": mismatches,
        "overall": {
            "successes": computed_successes,
            "total": computed_total,
            "rate": computed_successes / computed_total,
        },
        "suites": {
            suite: {
                "successes": values[0],
                "total": values[1],
                "rate": values[0] / values[1],
            }
            for suite, values in sorted(suite_totals.items())
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if mismatches or len(verified) != 40:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

```


## `select_paired_trials.py`

来源：`/root/wuqingman/analysis_exp7/select_paired_trials.py`

```python
"""Select matched LIBERO trial IDs with informative checkpoint outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(root: Path, task_name: str) -> dict[int, bool]:
    path = root / f"{task_name.replace(':', '_')}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["index"]): bool(item["success"]) for item in payload["episodes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="LABEL=/results/path")
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-pattern", type=int, default=2)
    args = parser.parse_args()

    runs = {label: Path(path) for label, path in (spec.split("=", 1) for spec in args.run)}
    cases = []
    for task_name in args.task:
        outcomes = {label: _load(root, task_name) for label, root in runs.items()}
        available = sorted(set.intersection(*(set(items) for items in outcomes.values() if items)))
        patterns: dict[str, list[int]] = {}
        for trial_id in available:
            pattern = ",".join(
                f"{label}={'S' if outcomes[label][trial_id] else 'F'}" for label in sorted(outcomes)
            )
            patterns.setdefault(pattern, []).append(trial_id)
        for pattern, trial_ids in sorted(patterns.items()):
            for trial_id in trial_ids[: args.per_pattern]:
                cases.append({"task_name": task_name, "trial_id": trial_id, "pattern": pattern})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(cases)} paired cases to {output}")
    for case in cases:
        print(case)


if __name__ == "__main__":
    main()

```


## `openloop_condition_diagnostics.py`

来源：`/root/wuqingman/analysis_exp7/openloop_condition_diagnostics.py`

```python
"""Task-targeted open-loop and conditioning diagnostics for LIBERO."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from data.ego_wam.branch import (
    EgoWamBranchDataConfig,
    build_branch_sample,
    build_receding_horizon_indices,
)
from data.ego_wam.samples import _build_episode_dict, _decode_frame, _resolve_image_keys
from data.lerobot_v3 import LeRobotV3Dataset
from egowam_native.eval.policy import build_policy


def _description(libero_src: Path, suite: str, task_id: int) -> str:
    sys.path.insert(0, str(libero_src))
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map

    return libero_task_map[suite][task_id].replace("_", " ")


def _episode_ids(ds, description: str, limit: int) -> list[int]:
    target = description.lower()
    matches = []
    for item in ds.episodes:
        tasks = [str(task).lower() for task in item.get("tasks", [])]
        if target in tasks:
            matches.append(int(item["episode_index"]))
        if len(matches) >= limit:
            break
    if not matches:
        raise ValueError(f"No episodes found for instruction {description!r}")
    return matches


def _frames(ds, episode_id: int, timestep: int, sample: dict, video_keys: dict) -> dict[str, np.ndarray]:
    result = {}
    for group, frame_index in zip(sample["image_keys"], sample["image_timesteps"]):
        if int(frame_index) != int(timestep):
            continue
        for key in group:
            image = _decode_frame(ds, episode_id, timestep, video_keys[key])
            result[key] = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--base", default="/root/wuqingman/models/SenseNova-U1-8B-MoT-SFT")
    parser.add_argument("--libero-src", default="/root/wuqingman/LIBERO-src")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--flow-seeds", default="101,202,303")
    parser.add_argument("--disable-gen-branch", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seeds = [int(value) for value in args.flow_seeds.split(",") if value]
    description = _description(Path(args.libero_src), args.suite, args.task_id)
    cfg = EgoWamBranchDataConfig(
        train_und=False,
        train_gen=True,
        train_act=True,
        action_horizon=16,
        action_exec_horizon=4,
        frame_sampling_stride=4,
        observation_image_keys=("image", "wrist_image"),
        include_robot_state=True,
        robot_state_history_window=1,
        robot_state_history_stride=1,
    )
    ds = LeRobotV3Dataset.open(args.root)
    image_keys = _resolve_image_keys(ds, cfg)
    episode_ids = _episode_ids(ds, description, args.num_episodes)
    policy = build_policy(
        checkpoint=args.checkpoint,
        model_name_or_path=args.base,
        tokenizer_path=args.base,
        action_horizon=16,
        action_dim=7,
        gpu_id=args.gpu_id,
        dtype="bfloat16",
        num_inference_steps=20,
        enable_gen_branch=not args.disable_gen_branch,
        robot_state_dim=8,
        robot_state_num_tokens=1,
        robot_state_injection_branch="und",
        normalize_action_state=True,
        action_state_stats_path=args.stats,
        action_state_norm_eps=1e-6,
        lang_hidden_size=1024,
        gen_hidden_size=1024,
        act_hidden_size=1024,
        flow_seed=seeds[0],
    )

    samples = []
    representative = None
    for episode_id in episode_ids:
        episode = _build_episode_dict(ds, episode_id, image_keys=image_keys)
        video_keys = episode["_lerobot_v3"]["video_keys"]
        count = len(ds.frame_rows(episode_id))
        all_timesteps = build_receding_horizon_indices(count, cfg.stride)
        positions = np.linspace(0, len(all_timesteps) - 1, args.num_frames, dtype=int)
        for position in positions:
            timestep = all_timesteps[int(position)]
            sample = build_branch_sample(episode, timestep=timestep, config=cfg)
            frames = _frames(ds, episode_id, timestep, sample, video_keys)
            state = np.asarray(sample["robot_state"], dtype=np.float32)
            gt = np.asarray(sample["action_chunk"], dtype=np.float32)
            mask = np.asarray(sample["action_mask"], dtype=bool)
            obs = {
                "instruction": episode["instruction"],
                "image": frames["image"],
                "wrist_image": frames["wrist_image"],
                "robot_state": state,
            }
            predictions = []
            for seed in seeds:
                policy.reset_flow_seed(seed)
                predictions.append(np.asarray(policy.predict_action_chunk(obs), dtype=np.float32))
            predictions_array = np.stack(predictions)
            mean_prediction = predictions_array.mean(axis=0)
            valid_prediction = mean_prediction[mask]
            valid_gt = gt[mask]
            samples.append(
                {
                    "episode_id": episode_id,
                    "timestep": timestep,
                    "mae": float(np.abs(valid_prediction - valid_gt).mean()),
                    "per_dim_mae": np.abs(valid_prediction - valid_gt).mean(axis=0).tolist(),
                    "gripper_accuracy": float(
                        ((valid_prediction[:, -1] >= 0.5) == (valid_gt[:, -1] >= 0.5)).mean()
                    ),
                    "flow_seed_std": float(predictions_array.std(axis=0).mean()),
                }
            )
            if representative is None and int(position) >= len(all_timesteps) // (2 * args.num_frames):
                representative = (obs, gt, mask)

    obs, gt, mask = representative
    zero_agent = np.zeros_like(obs["image"])
    zero_wrist = np.zeros_like(obs["wrist_image"])
    zero_state = np.zeros_like(obs["robot_state"])
    conditions = {
        "baseline": obs,
        "zero_agentview": {**obs, "image": zero_agent},
        "zero_wrist": {**obs, "wrist_image": zero_wrist},
        "zero_images": {**obs, "image": zero_agent, "wrist_image": zero_wrist},
        "zero_state": {**obs, "robot_state": zero_state},
        "wrong_instruction": {**obs, "instruction": "do nothing and keep the robot still"},
    }
    condition_predictions = {}
    for name, condition in conditions.items():
        policy.reset_flow_seed(seeds[0])
        condition_predictions[name] = np.asarray(
            policy.predict_action_chunk(condition), dtype=np.float32
        )
    baseline = condition_predictions["baseline"]
    ablations = {
        name: {
            "mean_abs_change": float(np.abs(prediction - baseline).mean()),
            "first_action": prediction[0].tolist(),
            "mae_vs_gt": float(np.abs(prediction[mask] - gt[mask]).mean()),
        }
        for name, prediction in condition_predictions.items()
    }
    payload = {
        "checkpoint": args.checkpoint_label,
        "suite": args.suite,
        "task_id": args.task_id,
        "description": description,
        "episode_ids": episode_ids,
        "flow_seeds": seeds,
        "summary": {
            "mean_mae": float(np.mean([item["mae"] for item in samples])),
            "mean_gripper_accuracy": float(
                np.mean([item["gripper_accuracy"] for item in samples])
            ),
            "mean_flow_seed_std": float(np.mean([item["flow_seed_std"] for item in samples])),
        },
        "samples": samples,
        "ablations": ablations,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

```


## `internal_activation_diagnostics.py`

来源：`/root/wuqingman/analysis_exp7/internal_activation_diagnostics.py`

```python
"""Capture internal activation and flow sensitivity for one LIBERO sample."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data.ego_wam.branch import EgoWamBranchDataConfig, build_branch_sample
from data.ego_wam.samples import _build_episode_dict, _decode_frame, _resolve_image_keys
from data.lerobot_v3 import LeRobotV3Dataset
from egowam_native.eval.policy import build_policy


def _description(libero_src: Path, suite: str, task_id: int) -> str:
    sys.path.insert(0, str(libero_src))
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map

    return libero_task_map[suite][task_id].replace("_", " ")


def _tensor_stats(value) -> list[dict]:
    values = value if isinstance(value, (tuple, list)) else (value,)
    result = []
    for index, item in enumerate(values):
        if not torch.is_tensor(item):
            continue
        tensor = item.detach().float()
        pooled = tensor.reshape(-1, tensor.shape[-1]).mean(dim=0)
        result.append(
            {
                "index": index,
                "shape": list(tensor.shape),
                "mean": float(tensor.mean().cpu()),
                "std": float(tensor.std().cpu()),
                "norm": float(tensor.norm().cpu()),
                "pooled": pooled.cpu().tolist(),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--base", default="/root/wuqingman/models/SenseNova-U1-8B-MoT-SFT")
    parser.add_argument("--libero-src", default="/root/wuqingman/LIBERO-src")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--flow-seed", type=int, default=101)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    description = _description(Path(args.libero_src), args.suite, args.task_id)
    cfg = EgoWamBranchDataConfig(
        train_und=False,
        train_gen=True,
        train_act=True,
        action_horizon=16,
        action_exec_horizon=4,
        frame_sampling_stride=4,
        observation_image_keys=("image", "wrist_image"),
        include_robot_state=True,
        robot_state_history_window=1,
        robot_state_history_stride=1,
    )
    ds = LeRobotV3Dataset.open(args.root)
    image_keys = _resolve_image_keys(ds, cfg)
    episode_id = next(
        int(item["episode_index"])
        for item in ds.episodes
        if description.lower() in [str(task).lower() for task in item.get("tasks", [])]
    )
    episode = _build_episode_dict(ds, episode_id, image_keys=image_keys)
    count = len(ds.frame_rows(episode_id))
    timestep = (count // 8) * 4
    sample = build_branch_sample(episode, timestep=timestep, config=cfg)
    video_keys = episode["_lerobot_v3"]["video_keys"]
    frames = {}
    for group, frame_index in zip(sample["image_keys"], sample["image_timesteps"]):
        if int(frame_index) != timestep:
            continue
        for key in group:
            frames[key] = np.asarray(
                _decode_frame(ds, episode_id, timestep, video_keys[key]).convert("RGB"),
                dtype=np.uint8,
            )
    state = np.asarray(sample["robot_state"], dtype=np.float32)
    baseline_obs = {
        "instruction": episode["instruction"],
        "image": frames["image"],
        "wrist_image": frames["wrist_image"],
        "robot_state": state,
    }
    policy = build_policy(
        checkpoint=args.checkpoint,
        model_name_or_path=args.base,
        tokenizer_path=args.base,
        action_horizon=16,
        action_dim=7,
        gpu_id=args.gpu_id,
        dtype="bfloat16",
        num_inference_steps=20,
        enable_gen_branch=True,
        robot_state_dim=8,
        robot_state_num_tokens=1,
        robot_state_injection_branch="und",
        normalize_action_state=True,
        action_state_stats_path=args.stats,
        action_state_norm_eps=1e-6,
        lang_hidden_size=1024,
        gen_hidden_size=1024,
        act_hidden_size=1024,
        flow_seed=args.flow_seed,
        record_flow_diagnostics=True,
    )
    model = policy.backend._get_model()
    captures: dict[str, list[list[dict]]] = defaultdict(list)

    def register(name, module):
        def hook(_module, _inputs, output):
            captures[name].append(_tensor_stats(output))

        return module.register_forward_hook(hook)

    layers = model.language_model.model.layers
    handles = [
        register("vision_patch", model.vision_model),
        register("robot_state_encoder", model.action_modules["robot_state_encoder"]),
        register("layer_0", layers[0]),
        register("layer_mid", layers[len(layers) // 2]),
        register("layer_last", layers[-1]),
        register("action_head", model.action_modules["action_head"]),
    ]
    conditions = {
        "baseline": baseline_obs,
        "zero_images": {
            **baseline_obs,
            "image": np.zeros_like(baseline_obs["image"]),
            "wrist_image": np.zeros_like(baseline_obs["wrist_image"]),
        },
        "zero_state": {**baseline_obs, "robot_state": np.zeros_like(state)},
        "wrong_instruction": {
            **baseline_obs,
            "instruction": "do nothing and keep the robot still",
        },
    }
    results = {}
    try:
        for condition_name, obs in conditions.items():
            captures.clear()
            policy.reset_flow_seed(args.flow_seed)
            action = np.asarray(policy.predict_action_chunk(obs), dtype=np.float32)
            diagnostics = policy.get_last_diagnostics()
            results[condition_name] = {
                "first_action": action[0].tolist(),
                "action_mean_abs": float(np.abs(action).mean()),
                "velocity_norms": diagnostics["velocity_norms"].tolist(),
                "activation_calls": dict(captures),
            }
    finally:
        for handle in handles:
            handle.remove()

    baseline_action = np.asarray(results["baseline"]["first_action"], dtype=np.float32)
    for condition_name, result in results.items():
        first_action = np.asarray(result["first_action"], dtype=np.float32)
        result["first_action_change_vs_baseline"] = float(
            np.abs(first_action - baseline_action).mean()
        )
    payload = {
        "checkpoint": args.checkpoint_label,
        "suite": args.suite,
        "task_id": args.task_id,
        "description": description,
        "episode_id": episode_id,
        "timestep": timestep,
        "flow_seed": args.flow_seed,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    for name, result in results.items():
        print(name, "action_change", result["first_action_change_vs_baseline"])


if __name__ == "__main__":
    main()

```


## `compare_activation_cosine.py`

来源：`/root/wuqingman/analysis_exp7/compare_activation_cosine.py`

```python
"""Compare pooled internal representations across checkpoints and tasks."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def _vector(payload: dict, condition: str, module: str, branch_index: int) -> np.ndarray:
    calls = payload["results"][condition]["activation_calls"][module]
    entries = calls[-1]
    entry = next(item for item in entries if int(item["index"]) == branch_index)
    return np.asarray(entry["pooled"], dtype=np.float32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.files]
    comparisons = []
    modules = ("layer_0", "layer_mid", "layer_last")
    for left, right in combinations(payloads, 2):
        left_label = f"{left['checkpoint']}/{left['suite']}:{left['task_id']}"
        right_label = f"{right['checkpoint']}/{right['suite']}:{right['task_id']}"
        row = {"left": left_label, "right": right_label}
        for module in modules:
            row[f"{module}_und_cosine"] = _cosine(
                _vector(left, "baseline", module, 0),
                _vector(right, "baseline", module, 0),
            )
            row[f"{module}_act_cosine"] = _cosine(
                _vector(left, "baseline", module, 2),
                _vector(right, "baseline", module, 2),
            )
        comparisons.append(row)
    condition_sensitivity = []
    for payload in payloads:
        label = f"{payload['checkpoint']}/{payload['suite']}:{payload['task_id']}"
        for condition in ("zero_images", "zero_state", "wrong_instruction"):
            row = {"label": label, "condition": condition}
            for module in modules:
                row[f"{module}_und_cosine"] = _cosine(
                    _vector(payload, "baseline", module, 0),
                    _vector(payload, condition, module, 0),
                )
                row[f"{module}_act_cosine"] = _cosine(
                    _vector(payload, "baseline", module, 2),
                    _vector(payload, condition, module, 2),
                )
            condition_sensitivity.append(row)
    result = {"cross_run": comparisons, "condition_sensitivity": condition_sensitivity}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

```


## `summarize_rollout_diagnostics.py`

来源：`/root/wuqingman/analysis_exp7/summarize_rollout_diagnostics.py`

```python
"""Summarize paired deterministic LIBERO rollout diagnostic NPZ files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _summary(path: Path) -> dict:
    data = np.load(path)
    actions = np.asarray(data["executed_actions"], dtype=np.float32)
    replans = np.asarray(data.get("step", []), dtype=np.int64)
    chunks = np.asarray(data.get("action_chunk_dataset", []), dtype=np.float32)
    states = np.asarray(data.get("robot_state", []), dtype=np.float32)
    flow_states = np.asarray(data.get("flow_states", []), dtype=np.float32)
    velocity_norms = np.asarray(data.get("velocity_norms", []), dtype=np.float32)
    gripper = actions[:, -1] if actions.size else np.asarray([])
    transitions = int(np.sum(gripper[1:] != gripper[:-1])) if len(gripper) > 1 else 0
    close_indices = np.flatnonzero(gripper > 0)
    return {
        "path": str(path),
        "success": bool(np.asarray(data["success"]).item()),
        "steps": int(np.asarray(data["steps"]).item()),
        "num_replans": int(len(replans)),
        "first_close_action": int(close_indices[0]) if len(close_indices) else None,
        "gripper_transitions": transitions,
        "close_fraction": float(np.mean(gripper > 0)) if len(gripper) else 0.0,
        "translation_abs_mean": float(np.abs(actions[:, :3]).mean()) if actions.size else 0.0,
        "rotation_abs_mean": float(np.abs(actions[:, 3:6]).mean()) if actions.size else 0.0,
        "chunk_translation_abs_mean": float(np.abs(chunks[..., :3]).mean()) if chunks.size else 0.0,
        "state_path_length": float(np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1).sum())
        if len(states) > 1
        else 0.0,
        "flow_initial_norm_mean": float(np.linalg.norm(flow_states[:, 0], axis=(-2, -1)).mean())
        if flow_states.size
        else 0.0,
        "flow_final_norm_mean": float(np.linalg.norm(flow_states[:, -1], axis=(-2, -1)).mean())
        if flow_states.size
        else 0.0,
        "velocity_norm_mean": float(velocity_norms.mean()) if velocity_norms.size else 0.0,
        "sample_seeds": np.asarray(data.get("sample_seed", []), dtype=np.int64).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summaries = [_summary(Path(path)) for path in args.files]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

```


## `gradient_conflict_diagnostics.py`

来源：`/root/wuqingman/analysis_exp7/gradient_conflict_diagnostics.py`

```python
"""Measure Generation-vs-Action gradient cosine on a real LIBERO batch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from data.ego_wam.branch import EgoWamBranchDataConfig, build_branch_sample
from data.ego_wam.dataloader import collate_packed
from data.ego_wam.native import _concat_image_views_horizontally, _pack_branch_sample
from data.ego_wam.samples import _build_episode_dict, _decode_frame, _resolve_image_keys
from data.lerobot_v3 import LeRobotV3Dataset
from egowam_native.action_state_normalization import ActionStateNormalizer
from egowam_native.eval.policy import TorchEgoWAMNativeBackend, _move_packed_batch_to_device


def _description(libero_src: Path, suite: str, task_id: int) -> str:
    sys.path.insert(0, str(libero_src))
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map

    return libero_task_map[suite][task_id].replace("_", " ")


def _group(name: str, layer_ids: set[int]) -> str | None:
    if name.startswith("vision_model."):
        return "vision_observation"
    if name.startswith("action_modules.robot_state_encoder."):
        return "state_encoder"
    if not name.startswith("language_model.model.layers."):
        return None
    layer_id = int(name.split(".")[3])
    if layer_id not in layer_ids:
        return None
    if "_mot_act" in name:
        return "selected_action_expert"
    if "_mot_gen" in name:
        return "selected_generation_expert"
    if "_mot_value" not in name:
        return "selected_shared_understanding"
    return None


def _metrics(image_grads, action_grads, groups, names) -> dict:
    accum = {}
    for image_grad, action_grad, group, name in zip(image_grads, action_grads, groups, names):
        if group is None:
            continue
        item = accum.setdefault(
            group,
            {
                "dot": torch.zeros((), device=action_grad.device if action_grad is not None else "cpu"),
                "image_sq": torch.zeros((), device=action_grad.device if action_grad is not None else "cpu"),
                "action_sq": torch.zeros((), device=action_grad.device if action_grad is not None else "cpu"),
                "used_parameters": 0,
                "total_parameters": 0,
                "examples": [],
            },
        )
        item["total_parameters"] += 1
        if image_grad is None or action_grad is None:
            continue
        image_float = image_grad.detach().float()
        action_float = action_grad.detach().float()
        item["dot"] = item["dot"].to(image_float.device) + (image_float * action_float).sum()
        item["image_sq"] = item["image_sq"].to(image_float.device) + image_float.square().sum()
        item["action_sq"] = item["action_sq"].to(image_float.device) + action_float.square().sum()
        item["used_parameters"] += 1
        if len(item["examples"]) < 5:
            item["examples"].append(name)
    result = {}
    for group, item in accum.items():
        image_norm = item["image_sq"].sqrt()
        action_norm = item["action_sq"].sqrt()
        cosine = item["dot"] / (image_norm * action_norm + 1e-12)
        result[group] = {
            "cosine": float(cosine.cpu()),
            "image_grad_norm": float(image_norm.cpu()),
            "action_grad_norm": float(action_norm.cpu()),
            "used_parameters": item["used_parameters"],
            "total_parameters": item["total_parameters"],
            "examples": item["examples"],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--base", default="/root/wuqingman/models/SenseNova-U1-8B-MoT-SFT")
    parser.add_argument("--libero-src", default="/root/wuqingman/LIBERO-src")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    description = _description(Path(args.libero_src), args.suite, args.task_id)
    cfg = EgoWamBranchDataConfig(
        train_und=False,
        train_gen=True,
        train_act=True,
        action_horizon=16,
        action_exec_horizon=4,
        frame_sampling_stride=4,
        observation_image_keys=("image", "wrist_image"),
        include_robot_state=True,
        robot_state_history_window=1,
        robot_state_history_stride=1,
    )
    ds = LeRobotV3Dataset.open(args.root)
    image_keys = _resolve_image_keys(ds, cfg)
    episode_id = next(
        int(item["episode_index"])
        for item in ds.episodes
        if description.lower() in [str(task).lower() for task in item.get("tasks", [])]
    )
    episode = _build_episode_dict(ds, episode_id, image_keys=image_keys)
    timestep = (len(ds.frame_rows(episode_id)) // 8) * 4
    sample = build_branch_sample(episode, timestep=timestep, config=cfg)
    video_keys = episode["_lerobot_v3"]["video_keys"]
    image_pil = []
    for group, frame_index in zip(sample["image_keys"], sample["image_timesteps"]):
        views = [_decode_frame(ds, episode_id, frame_index, video_keys[key]) for key in group]
        image_pil.append(_concat_image_views_horizontally(views))
    sample["image_pil"] = image_pil

    backend = TorchEgoWAMNativeBackend(
        checkpoint=args.checkpoint,
        model_name_or_path=args.base,
        tokenizer_path=args.base,
        action_horizon=16,
        action_dim=7,
        device=f"cuda:{args.gpu_id}",
        dtype="bfloat16",
        num_inference_steps=20,
        enable_gen_branch=True,
        robot_state_dim=8,
        robot_state_num_tokens=1,
        robot_state_injection_branch="und",
        normalize_action_state=True,
        action_state_stats_path=args.stats,
        lang_hidden_size=1024,
        gen_hidden_size=1024,
        act_hidden_size=1024,
    )
    model = backend._get_model()
    model.disable_gen_forward = False
    model.image_gen_loss_weight = 1.0
    model.action_gen_loss_weight = 1.0
    tokenizer = backend._get_tokenizer()
    normalizer = ActionStateNormalizer.from_stats_path(args.stats, enabled=True, eps=1e-6)
    packed = _pack_branch_sample(
        sample,
        tokenizer=tokenizer,
        model_config=model.config,
        max_tokens=16384,
        action_state_normalizer=normalizer,
    )
    batch = collate_packed([packed])
    device = torch.device(f"cuda:{args.gpu_id}")
    batch = _move_packed_batch_to_device(batch, device, non_blocking=False)

    torch.manual_seed(101)
    output = model(**batch)
    if output.image_gen_loss is None or output.action_gen_loss is None:
        raise RuntimeError("Expected both image_gen_loss and action_gen_loss")
    layer_ids = {0, len(model.language_model.model.layers) // 2, len(model.language_model.model.layers) - 1}
    selected = [
        (name, parameter, _group(name, layer_ids))
        for name, parameter in model.named_parameters()
        if _group(name, layer_ids) is not None and parameter.requires_grad
    ]
    names = [name for name, _, _ in selected]
    parameters = [parameter for _, parameter, _ in selected]
    groups = [group for _, _, group in selected]
    image_grads = torch.autograd.grad(
        output.image_gen_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    action_grads = torch.autograd.grad(
        output.action_gen_loss,
        parameters,
        allow_unused=True,
    )
    payload = {
        "checkpoint": args.checkpoint_label,
        "suite": args.suite,
        "task_id": args.task_id,
        "description": description,
        "episode_id": episode_id,
        "timestep": timestep,
        "image_gen_loss": float(output.image_gen_loss.detach().cpu()),
        "action_gen_loss": float(output.action_gen_loss.detach().cpu()),
        "gradient_groups": _metrics(image_grads, action_grads, groups, names),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

```


## `synthesize_findings.py`

来源：`/root/wuqingman/analysis_exp7/synthesize_findings.py`

```python
"""Synthesize task, rollout, open-loop and internal diagnostic artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsons(directory: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
        if path.is_file()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.analysis_dir)
    task_rows = json.loads((root / "task_profile.json").read_text(encoding="utf-8"))
    openloop = _read_jsons(root / "openloop")
    internal = _read_jsons(root / "internal")
    cosine_path = root / "internal" / "activation_cosine.json"
    cosine = json.loads(cosine_path.read_text(encoding="utf-8")) if cosine_path.exists() else {}
    rollout_files = sorted((root / "paired_rollouts").glob("*/*_diagnostics.npz"))

    lines = ["# Experiment 7 failure-analysis findings", ""]
    lines.extend(
        [
            "## Task-level evidence",
            "",
            "| Checkpoint | Suite/task | Success | Description |",
            "|---|---|---:|---|",
        ]
    )
    for checkpoint in sorted({row["checkpoint"] for row in task_rows}):
        checkpoint_rows = [row for row in task_rows if row["checkpoint"] == checkpoint]
        for row in sorted(checkpoint_rows, key=lambda item: item["success_rate"])[:8]:
            lines.append(
                f"| {checkpoint} | {row['task_name']} | {100 * row['success_rate']:.1f}% | "
                f"{row['description']} |"
            )

    lines.extend(
        [
            "",
            "## Open-loop and condition ablations",
            "",
            "| Checkpoint/task | MAE | Gripper acc | Flow-seed std | Zero image Δ | "
            "Zero state Δ | Wrong instruction Δ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in openloop:
        summary = item["summary"]
        ablations = item["ablations"]
        lines.append(
            f"| {item['checkpoint']}/{item['suite']}:{item['task_id']} | "
            f"{summary['mean_mae']:.4f} | {100 * summary['mean_gripper_accuracy']:.1f}% | "
            f"{summary['mean_flow_seed_std']:.4f} | "
            f"{ablations['zero_images']['mean_abs_change']:.4f} | "
            f"{ablations['zero_state']['mean_abs_change']:.4f} | "
            f"{ablations['wrong_instruction']['mean_abs_change']:.4f} |"
        )

    lines.extend(["", "## Paired deterministic rollouts", ""])
    if rollout_files:
        lines.append(f"Captured diagnostic rollouts: {len(rollout_files)}")
        lines.append("")
        lines.extend(
            [
                "| File | Success | Steps | Gripper transitions |",
                "|---|---:|---:|---:|",
            ]
        )
        import numpy as np

        for path in rollout_files:
            data = np.load(path)
            actions = data["executed_actions"]
            transitions = int((actions[1:, -1] != actions[:-1, -1]).sum()) if len(actions) > 1 else 0
            lines.append(
                f"| {path.parent.name}/{path.stem} | {bool(data['success'])} | "
                f"{int(data['steps'])} | {transitions} |"
            )
    else:
        lines.append("No rollout diagnostics yet.")

    lines.extend(["", "## Internal activation diagnostics", ""])
    for item in internal:
        if "results" not in item:
            continue
        lines.append(
            f"- {item['checkpoint']} {item['suite']}:{item['task_id']}: "
            + ", ".join(
                f"{name} actionΔ={result['first_action_change_vs_baseline']:.4f}"
                for name, result in item["results"].items()
                if name != "baseline"
            )
        )

    if cosine:
        lines.extend(
            [
                "",
                "### Pooled activation cosine",
                "",
                "| Left | Right | Last Und cosine | Last Act cosine |",
                "|---|---|---:|---:|",
            ]
        )
        for row in cosine["cross_run"]:
            lines.append(
                f"| {row['left']} | {row['right']} | {row['layer_last_und_cosine']:.4f} | "
                f"{row['layer_last_act_cosine']:.4f} |"
            )
        lines.extend(
            [
                "",
                "| Run | Condition | Last Und cosine | Last Act cosine |",
                "|---|---|---:|---:|",
            ]
        )
        for row in cosine["condition_sensitivity"]:
            lines.append(
                f"| {row['label']} | {row['condition']} | {row['layer_last_und_cosine']:.4f} | "
                f"{row['layer_last_act_cosine']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Evidence-based interpretation checklist",
            "",
            "- Low closed-loop success with low open-loop MAE points to compounding error/control recovery, "
            "not failure to fit demonstrations.",
            "- Large gripper-transition counts point to unstable grasp/recovery decisions.",
            "- Large cross-checkpoint swings on the same task point to checkpoint specialization or "
            "forgetting rather than intrinsic task impossibility.",
            "- Large fixed-noise image/state/instruction ablation changes identify which conditioning "
            "path actually drives the action.",
            "- Large flow-seed variance means paired checkpoint comparisons must use identical action-noise "
            "sequences.",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

```


## `export_success_dashboard_html.py`

来源：`/root/wuqingman/analysis_exp7/export_success_dashboard_html.py`

```python
"""Export the LIBERO task profile as a standalone, dependency-free HTML dashboard."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>实验7 · LIBERO成功率总览</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg:#111318; --panel:#191c23; --panel2:#21252e; --text:#edf0f7;
      --muted:#9da6b7; --line:#343a46; --c80:#6ea8fe; --c90:#59d499;
      --c100:#d7a7ff; --warn:#f0b35a; --bad:#f06f79; --good:#59d499;
    }
    @media (prefers-color-scheme: light) {
      :root {
        --bg:#f7f8fb; --panel:#fff; --panel2:#f1f3f7; --text:#20242c;
        --muted:#667085; --line:#d9dde6;
      }
    }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5 system-ui,sans-serif; }
    main { max-width:1440px; margin:auto; padding:28px; }
    h1 { margin:0 0 4px; font-size:26px; }
    h2 { margin:28px 0 10px; font-size:19px; }
    .muted { color:var(--muted); }
    .grid3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; }
    .stat-value { font-size:27px; font-weight:700; }
    .stat-label { color:var(--muted); margin-top:3px; }
    .callout { margin:16px 0; padding:13px 16px; border-left:4px solid var(--c90); background:var(--panel2); }
    .controls { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }
    select,button { color:var(--text); background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:7px 10px; }
    label { display:flex; align-items:center; gap:7px; color:var(--muted); }
    .legend { display:flex; gap:16px; margin:8px 0 14px; }
    .legend span::before { content:""; display:inline-block; width:10px; height:10px; margin-right:5px; border-radius:2px; background:var(--color); }
    .chart { display:grid; gap:11px; }
    .chart-row { display:grid; grid-template-columns:150px 1fr; align-items:center; gap:10px; }
    .task-chart .chart-row { grid-template-columns:48px 1fr; }
    .bars { display:grid; gap:4px; }
    .bar-line { display:grid; grid-template-columns:46px 1fr 52px; align-items:center; gap:7px; }
    .track { height:12px; background:var(--panel2); border-radius:3px; overflow:hidden; }
    .bar { height:100%; width:calc(var(--value) * 1%); background:var(--color); }
    .pct { text-align:right; font-variant-numeric:tabular-nums; }
    table { width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); }
    th,td { padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
    th { text-align:left; position:sticky; top:0; background:var(--panel2); }
    td.num,th.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
    tr.bad td:first-child { border-left:4px solid var(--bad); }
    tr.warn td:first-child { border-left:4px solid var(--warn); }
    tr.good td:first-child { border-left:4px solid var(--good); }
    .scroll { max-height:620px; overflow:auto; border-radius:8px; }
    .caption { margin-top:9px; color:var(--muted); font-size:12px; }
    code { background:var(--panel2); padding:2px 5px; border-radius:4px; }
    @media (max-width:760px) {
      main { padding:16px; } .grid3 { grid-template-columns:1fr; }
      .chart-row { grid-template-columns:90px 1fr; }
    }
    @media print {
      :root { --bg:#fff; --panel:#fff; --panel2:#f5f5f5; --text:#111; --muted:#555; --line:#ccc; }
      main { max-width:none; padding:0; } .controls { display:none; } .scroll { max-height:none; overflow:visible; }
      .card,table { break-inside:avoid; }
    }
  </style>
</head>
<body>
<main>
  <h1>实验7 · LIBERO成功率总览</h1>
  <div class="muted">Gen+Act权重 · 40个任务 · 每个任务50次评测 · replan 10 · MuJoCo 3.3.2</div>

  <div class="grid3" id="overall" style="margin-top:18px"></div>
  <div class="callout"><b>已通过三层校验。</b> summary.csv、40个task JSON和逐trial日志逐项一致；120/120个task无mismatch。</div>

  <h2>各Suite成功率</h2>
  <div class="card">
    <div class="legend"></div>
    <div class="chart" id="suiteChart"></div>
    <div class="caption">横轴：成功率0–100% · 数据来源：正式评测，2026年7月完成。</div>
  </div>

  <div class="controls" style="margin-top:28px">
    <h2 style="margin:0">逐Task成功率对比</h2>
    <div style="display:flex;gap:12px;align-items:center">
      <select id="suiteSelect"></select>
      <label><input id="sortWorst" type="checkbox" /> 最难任务优先排序</label>
    </div>
  </div>
  <div class="grid3" id="suiteStats" style="margin:14px 0"></div>
  <div class="card">
    <div class="legend"></div>
    <div class="chart task-chart" id="taskChart"></div>
  </div>
  <div class="scroll" style="margin-top:14px">
    <table>
      <thead><tr><th>Task</th><th>原始LIBERO任务名称</th><th class="num">80k</th><th class="num">90k</th><th class="num">100k</th><th class="num">最佳 / 跨度</th></tr></thead>
      <tbody id="taskTable"></tbody>
    </table>
  </div>

  <h2>权重之间变化最大的任务</h2>
  <p class="muted">80k、90k和100k之间成功率变化最大的任务。大幅波动揭示总体平均值掩盖的任务干扰、遗忘或checkpoint特异性。</p>
  <div class="scroll">
    <table>
      <thead><tr><th>Suite / Task</th><th>原始LIBERO任务名称</th><th class="num">最低</th><th class="num">最高</th><th class="num">跨度</th></tr></thead>
      <tbody id="shiftTable"></tbody>
    </table>
  </div>
  <p class="caption">红色：三个权重均低于30%。黄色：至少一个权重低于20%，或权重间差距≥40个百分点。可使用浏览器“打印”导出PDF。</p>
</main>
<script>
const DATA = __DATA__;
const CPS = ["80k","90k","100k"];
const COLORS = {"80k":"var(--c80)","90k":"var(--c90)","100k":"var(--c100)"};
const suiteNames = {libero_spatial:"LIBERO-Spatial",libero_object:"LIBERO-Object",libero_goal:"LIBERO-Goal",libero_10:"LIBERO-10"};
const bySuite = {};
for (const row of DATA.rows) {
  bySuite[row.suite] ??= {};
  bySuite[row.suite][row.task_id] ??= {description:row.description,rates:{}};
  bySuite[row.suite][row.task_id].rates[row.checkpoint] = row.success_rate * 100;
}
function legendHtml() {
  return CPS.map(cp=>`<span style="--color:${COLORS[cp]}">${cp}</span>`).join("");
}
document.querySelectorAll(".legend").forEach(el=>el.innerHTML=legendHtml());
document.getElementById("overall").innerHTML = CPS.map(cp=>
  `<div class="card"><div class="stat-value" style="color:${COLORS[cp]}">${DATA.overall[cp].rate}%</div><div class="stat-label">${cp} 总体 · ${DATA.overall[cp].successes}/2000</div></div>`
).join("");
function bars(values) {
  return `<div class="bars">${CPS.map(cp=>`<div class="bar-line"><b>${cp}</b><div class="track"><div class="bar" style="--value:${values[cp]};--color:${COLORS[cp]}"></div></div><span class="pct">${values[cp].toFixed(1)}%</span></div>`).join("")}</div>`;
}
document.getElementById("suiteChart").innerHTML = Object.keys(suiteNames).map(suite=>
  `<div class="chart-row"><b>${suiteNames[suite]}</b>${bars(DATA.suiteTotals[suite])}</div>`
).join("");
const select = document.getElementById("suiteSelect");
select.innerHTML = Object.keys(suiteNames).map(s=>`<option value="${s}">${suiteNames[s]}</option>`).join("");
function rowClass(values) {
  const vals=Object.values(values), max=Math.max(...vals), min=Math.min(...vals);
  if(max<30)return"bad"; if(min<20||max-min>=40)return"warn"; if(min>=80)return"good"; return"";
}
function renderTasks() {
  const suite=select.value||"libero_spatial", tasks=Object.entries(bySuite[suite]).map(([id,v])=>({id:+id,...v}));
  if(document.getElementById("sortWorst").checked)tasks.sort((a,b)=>Object.values(a.rates).reduce((x,y)=>x+y,0)-Object.values(b.rates).reduce((x,y)=>x+y,0)); else tasks.sort((a,b)=>a.id-b.id);
  document.getElementById("suiteStats").innerHTML=CPS.map(cp=>`<div class="card"><div class="stat-value" style="color:${COLORS[cp]}">${DATA.suiteTotals[suite][cp].toFixed(1)}%</div><div class="stat-label">${cp} · ${suiteNames[suite]}</div></div>`).join("");
  document.getElementById("taskChart").innerHTML=tasks.map(t=>`<div class="chart-row"><b>T${t.id}</b>${bars(t.rates)}</div>`).join("");
  document.getElementById("taskTable").innerHTML=tasks.map(t=>{const vals=CPS.map(cp=>t.rates[cp]),max=Math.max(...vals),min=Math.min(...vals),best=CPS[vals.indexOf(max)];return`<tr class="${rowClass(t.rates)}"><td>T${t.id}</td><td>${t.description}</td>${CPS.map(cp=>`<td class="num">${t.rates[cp].toFixed(0)}%</td>`).join("")}<td class="num">${best} · +${(max-min).toFixed(0)} pp</td></tr>`}).join("");
}
select.value="libero_spatial"; select.onchange=renderTasks; document.getElementById("sortWorst").onchange=renderTasks; renderTasks();
const shifts=[];
for(const [suite,tasks] of Object.entries(bySuite))for(const [id,t] of Object.entries(tasks)){const vals=CPS.map(cp=>t.rates[cp]),max=Math.max(...vals),min=Math.min(...vals);shifts.push({suite,id,description:t.description,min,max,worst:CPS[vals.indexOf(min)],best:CPS[vals.indexOf(max)],range:max-min});}
shifts.sort((a,b)=>b.range-a.range);
document.getElementById("shiftTable").innerHTML=shifts.slice(0,15).map(x=>`<tr class="${x.range>=50?"bad":"warn"}"><td>${suiteNames[x.suite]} · T${x.id}</td><td>${x.description}</td><td class="num">${x.worst} · ${x.min.toFixed(0)}%</td><td class="num">${x.best} · ${x.max.toFixed(0)}%</td><td class="num">${x.range.toFixed(0)} pp</td></tr>`).join("");
</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    checkpoints = ("80k", "90k", "100k")
    overall = {}
    suite_totals: dict[str, dict[str, float]] = defaultdict(dict)
    for checkpoint in checkpoints:
        cp_rows = [row for row in rows if row["checkpoint"] == checkpoint]
        successes = sum(row["successes"] for row in cp_rows)
        total = sum(row["total"] for row in cp_rows)
        overall[checkpoint] = {
            "successes": successes,
            "total": total,
            "rate": round(100 * successes / total, 2),
        }
        for suite in sorted({row["suite"] for row in cp_rows}):
            suite_rows = [row for row in cp_rows if row["suite"] == suite]
            suite_totals[suite][checkpoint] = 100 * sum(
                row["successes"] for row in suite_rows
            ) / sum(row["total"] for row in suite_rows)
    payload = {"rows": rows, "overall": overall, "suiteTotals": suite_totals}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False)),
        encoding="utf-8",
    )
    print(f"wrote {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

```


## `run_exp7_libero_4suite_eval.sh`

来源：`/root/wuqingman/run_exp7_libero_4suite_eval.sh`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

# Formal LIBERO evaluation for Experiment 7 (Generation + Action).
# Usage: bash run_exp7_libero_4suite_eval.sh <step> [num_trials] [gpus_csv]
#
# Both checkpoints may run concurrently with the same GPUs. Each process loads
# one model per GPU, while MuJoCo renders through EGL.

ROOT=/root/wuqingman
JOB=7-0712_libero_4suite_gen_act_stable_100k
TRAIN="$ROOT/Ego-WAM-egowam-stable/training"
LIBBIN="$ROOT/venvs/egowam-libero-mj332/bin"
BASE="$ROOT/models/SenseNova-U1-8B-MoT-SFT"
RUN="$ROOT/RUN/$JOB"
TASKDIR="$ROOT/eval_tasks_libero"
STATSDIR=/mnt/public/wuqingman/libero_eval/eval_stats
EGL_VENDOR="$ROOT/egl_vendor/10_nvidia.json"

STEP="${1:?usage: $0 <step> [num_trials] [gpus_csv]}"
NUM_TRIALS="${2:-50}"
GPUS_CSV="${3:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

CKPT="$RUN/step_${STEP}_hf"
OUT="$RUN/eval_mj332_step${STEP}_${NUM_TRIALS}trials_4suite_replan10"
LOGDIR="$RUN/eval_logs/mj332_step${STEP}_${NUM_TRIALS}trials_4suite_replan10"

test -s "$CKPT/model.safetensors"
test -x "$LIBBIN/python"
test -f "$BASE/model.safetensors.index.json"
test -f "$EGL_VENDOR"
mkdir -p "$OUT" "$LOGDIR"

cd "$TRAIN"

run_one() {
  local task="$1"
  local gpu="$2"
  local suite="${task%%:*}"
  local safe="${task//:/_}"
  local stats="$STATSDIR/$suite/stats.json"
  test -f "$stats"

  env \
    -u CUDA_VISIBLE_DEVICES \
    -u MUJOCO_EGL_DEVICE_ID \
    PATH="$LIBBIN:$PATH" \
    PYTHONPATH="$TRAIN" \
    TOKENIZERS_PARALLELISM=false \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID=0 \
    __EGL_VENDOR_LIBRARY_FILENAMES="$EGL_VENDOR" \
    DISPLAY= \
    XDG_RUNTIME_DIR=/tmp \
    "$LIBBIN/python" -m egowam_native.eval.libero_rollout \
      --task-name "$task" \
      --checkpoint "$CKPT" \
      --model-name-or-path "$BASE" \
      --tokenizer-path "$BASE" \
      --output-dir "$OUT" \
      --gpu-id "$gpu" \
      --num-trials "$NUM_TRIALS" \
      --seed 0 \
      --action-dim 7 \
      --action-horizon 16 \
      --action-exec-horizon 10 \
      --num-wait-steps 10 \
      --dtype bfloat16 \
      --num-inference-steps 20 \
      --robot-state-dim 8 \
      --robot-state-num-tokens 1 \
      --robot-state-injection-branch und \
      --normalize-action-state \
      --action-state-stats-path "$stats" \
      --action-state-norm-eps 1e-6 \
      --lang-hidden-size 1024 \
      --gen-hidden-size 1024 \
      --act-hidden-size 1024 \
      > "$LOGDIR/${safe}.log" 2>&1
}

for suite in libero_spatial libero_object libero_goal libero_10; do
  task_file="$TASKDIR/$suite.txt"
  test -f "$task_file"
  mapfile -t TASKS < <(grep -vE '^[[:space:]]*(#|$)' "$task_file")
  echo "[suite] step=$STEP suite=$suite tasks=${#TASKS[@]} trials=$NUM_TRIALS"

  i=0
  while [[ $i -lt ${#TASKS[@]} ]]; do
    pids=()
    labels=()
    for ((j=0; j<${#GPUS[@]} && i<${#TASKS[@]}; j++, i++)); do
      task="${TASKS[$i]}"
      gpu="${GPUS[$j]}"
      echo "[launch] step=$STEP $task -> gpu $gpu"
      run_one "$task" "$gpu" &
      pids+=("$!")
      labels+=("$task")
    done
    for ((j=0; j<${#pids[@]}; j++)); do
      if ! wait "${pids[$j]}"; then
        echo "[ERROR] step=$STEP task=${labels[$j]} failed; see $LOGDIR" >&2
        exit 1
      fi
    done
    echo "[wave] step=$STEP suite=$suite completed=$i/${#TASKS[@]}"
  done
done

env PATH="$LIBBIN:$PATH" PYTHONPATH="$TRAIN" \
  "$LIBBIN/python" - "$OUT" <<'PY'
import sys
from egowam_native.eval.common import load_task_results, write_summary

out = sys.argv[1]
results = load_task_results(out)
if len(results) != 40:
    raise RuntimeError(f"Expected 40 task results, found {len(results)} in {out}")
write_summary(results, out)
print(f"[summary] wrote {out}/summary.csv")
PY

echo "[DONE] step=$STEP output=$OUT/summary.csv"

```


# 附录B：Ego-WAM核心诊断改动Diff

FastWAM AI应借鉴方法，不应机械套用模块名。

```diff
diff --git a/training/egowam_native/eval/libero_rollout.py b/training/egowam_native/eval/libero_rollout.py
index 178ca19..ee4f844 100644
--- a/training/egowam_native/eval/libero_rollout.py
+++ b/training/egowam_native/eval/libero_rollout.py
@@ -36,6 +36,42 @@ def libero_policy_action_to_env(action) -> np.ndarray:
     return converted
 
 
+class LiberoGripperHysteresis:
+    """Convert dataset-space gripper values with hysteresis and minimum hold."""
+
+    def __init__(
+        self,
+        *,
+        close_threshold: float = 0.35,
+        open_threshold: float = 0.65,
+        min_hold_steps: int = 10,
+        initially_open: bool = True,
+    ):
+        if close_threshold >= open_threshold:
+            raise ValueError("close_threshold must be smaller than open_threshold")
+        self.close_threshold = float(close_threshold)
+        self.open_threshold = float(open_threshold)
+        self.min_hold_steps = max(0, int(min_hold_steps))
+        self.is_open = bool(initially_open)
+        self.steps_since_transition = self.min_hold_steps
+
+    def convert(self, action) -> np.ndarray:
+        converted = np.asarray(action, dtype=np.float32).copy()
+        value = float(converted[-1])
+        desired_open = self.is_open
+        if value >= self.open_threshold:
+            desired_open = True
+        elif value <= self.close_threshold:
+            desired_open = False
+        if desired_open != self.is_open and self.steps_since_transition >= self.min_hold_steps:
+            self.is_open = desired_open
+            self.steps_since_transition = 0
+        else:
+            self.steps_since_transition += 1
+        converted[-1] = -1.0 if self.is_open else 1.0
+        return converted
+
+
 def configure_libero_render_meshes(env) -> None:
     """Render visual meshes only, including under the MuJoCo 2.3 / NumPy 2 stride bug."""
     context = env.sim._render_context_offscreen
@@ -70,11 +106,28 @@ def _quat_xyzw_to_rotvec(quat) -> np.ndarray:
     return (quat[:3] * (2.0 * np.arccos(w)) / den).astype(np.float32)
 
 
-def libero_obs_to_policy_obs(obs: dict, instruction: str) -> dict:
-    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
+def _transform_libero_image(image, mode: str) -> np.ndarray:
+    array = np.asarray(image)
+    if mode == "rotate180":
+        array = array[::-1, ::-1]
+    elif mode == "vertical":
+        array = array[::-1]
+    elif mode == "horizontal":
+        array = array[:, ::-1]
+    elif mode != "none":
+        raise ValueError(f"Unsupported LIBERO image transform: {mode}")
+    return np.ascontiguousarray(array)
+
+
+def libero_obs_to_policy_obs(
+    obs: dict,
+    instruction: str,
+    image_transform: str = "rotate180",
+) -> dict:
+    image = _transform_libero_image(obs["agentview_image"], image_transform)
     wrist = None
     if "robot0_eye_in_hand_image" in obs:
-        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
+        wrist = _transform_libero_image(obs["robot0_eye_in_hand_image"], image_transform)
     state = None
     state_keys = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
     if any(key in obs for key in state_keys):
@@ -117,6 +170,11 @@ def run_episode(
     action_exec_horizon: int,
     num_wait_steps: int,
     use_action_ensemble: bool = False,
+    use_gripper_hysteresis: bool = False,
+    gripper_close_threshold: float = 0.35,
+    gripper_open_threshold: float = 0.65,
+    gripper_min_hold_steps: int = 10,
+    image_transform: str = "rotate180",
 ) -> EpisodeResult:
     env.reset()
     configure_libero_render_meshes(env)
@@ -126,20 +184,36 @@ def run_episode(
     steps = 0
     pending_actions: list[list[float]] = []
     ensembler = ActionEnsembler() if use_action_ensemble else None
+    gripper_controller = (
+        LiberoGripperHysteresis(
+            close_threshold=gripper_close_threshold,
+            open_threshold=gripper_open_threshold,
+            min_hold_steps=gripper_min_hold_steps,
+        )
+        if use_gripper_hysteresis
+        else None
+    )
     while steps < max_steps + num_wait_steps:
         if steps < num_wait_steps:
             obs, _, done, _ = env.step(get_libero_dummy_action())
             steps += 1
             continue
         if not pending_actions:
-            action_chunk = policy.predict_action_chunk(libero_obs_to_policy_obs(obs, instruction))
+            action_chunk = policy.predict_action_chunk(
+                libero_obs_to_policy_obs(obs, instruction, image_transform)
+            )
             pending_actions = build_pending_actions(
                 action_chunk,
                 start_timestamp=steps,
                 action_exec_horizon=action_exec_horizon,
                 ensembler=ensembler,
             )
-        action = libero_policy_action_to_env(pending_actions.pop(0))
+        dataset_action = pending_actions.pop(0)
+        action = (
+            gripper_controller.convert(dataset_action)
+            if gripper_controller is not None
+            else libero_policy_action_to_env(dataset_action)
+        )
         obs, _, done, _ = env.step(action)
         steps += 1
         if done:
@@ -197,6 +271,15 @@ def parse_args(argv=None):
     parser.add_argument("--action-horizon", type=int, default=16)
     parser.add_argument("--action-exec-horizon", type=int, default=4)
     parser.add_argument("--use-action-ensemble", action="store_true")
+    parser.add_argument("--use-gripper-hysteresis", action="store_true")
+    parser.add_argument("--gripper-close-threshold", type=float, default=0.35)
+    parser.add_argument("--gripper-open-threshold", type=float, default=0.65)
+    parser.add_argument("--gripper-min-hold-steps", type=int, default=10)
+    parser.add_argument(
+        "--image-transform",
+        choices=("rotate180", "vertical", "horizontal", "none"),
+        default="rotate180",
+    )
     parser.add_argument("--num-wait-steps", type=int, default=5)
     parser.add_argument("--seed", type=int, default=0)
     parser.add_argument("--dtype", default="bfloat16")
@@ -256,6 +339,11 @@ def main(argv=None):
             action_exec_horizon=args.action_exec_horizon,
             num_wait_steps=args.num_wait_steps,
             use_action_ensemble=args.use_action_ensemble,
+            use_gripper_hysteresis=args.use_gripper_hysteresis,
+            gripper_close_threshold=args.gripper_close_threshold,
+            gripper_open_threshold=args.gripper_open_threshold,
+            gripper_min_hold_steps=args.gripper_min_hold_steps,
+            image_transform=args.image_transform,
         )
         episodes.append(EpisodeResult(index=idx, success=episode.success, steps=episode.steps, info=episode.info))
         print(f"[libero] {args.task_name} trial={idx} success={episode.success}", flush=True)
diff --git a/training/egowam_native/eval/libero_vis.py b/training/egowam_native/eval/libero_vis.py
index ee18092..6d02655 100644
--- a/training/egowam_native/eval/libero_vis.py
+++ b/training/egowam_native/eval/libero_vis.py
@@ -11,6 +11,7 @@ import numpy as np
 from .action_ensembler import ActionEnsembler, build_pending_actions
 from .common import safe_task_filename
 from .libero_rollout import (
+    LiberoGripperHysteresis,
     _get_max_steps,
     build_libero_env,
     configure_libero_render_meshes,
@@ -38,9 +39,15 @@ def record_episode(
     num_wait_steps: int,
     video_path: Path,
     action_path: Path | None,
+    diagnostic_path: Path | None,
     camera: str,
     fps: int,
     use_action_ensemble: bool = False,
+    use_gripper_hysteresis: bool = False,
+    gripper_close_threshold: float = 0.35,
+    gripper_open_threshold: float = 0.65,
+    gripper_min_hold_steps: int = 10,
+    image_transform: str = "rotate180",
     show_progress: bool = True,
 ) -> tuple[bool, int, int]:
     env.reset()
@@ -51,7 +58,17 @@ def record_episode(
     steps = 0
     pending_actions: list[list[float]] = []
     recorded_actions: list[np.ndarray] = []
+    diagnostic_records: list[dict] = []
     ensembler = ActionEnsembler() if use_action_ensemble else None
+    gripper_controller = (
+        LiberoGripperHysteresis(
+            close_threshold=gripper_close_threshold,
+            open_threshold=gripper_open_threshold,
+            min_hold_steps=gripper_min_hold_steps,
+        )
+        if use_gripper_hysteresis
+        else None
+    )
 
     import imageio
 
@@ -70,14 +87,40 @@ def record_episode(
                 action = get_libero_dummy_action()
             else:
                 if not pending_actions:
-                    action_chunk = policy.predict_action_chunk(libero_obs_to_policy_obs(obs, instruction))
+                    policy_obs = libero_obs_to_policy_obs(obs, instruction, image_transform)
+                    action_chunk = policy.predict_action_chunk(policy_obs)
+                    if diagnostic_path is not None:
+                        model_diag = policy.get_last_diagnostics()
+                        diagnostic_records.append(
+                            {
+                                "step": steps,
+                                "image": np.asarray(policy_obs["image"], dtype=np.uint8),
+                                "wrist_image": np.asarray(policy_obs["wrist_image"], dtype=np.uint8),
+                                "robot_state": np.asarray(policy_obs["robot_state"], dtype=np.float32),
+                                "action_chunk_dataset": np.asarray(action_chunk, dtype=np.float32),
+                                "action_chunk_env": libero_policy_action_to_env(action_chunk),
+                                "sample_seed": int(model_diag.get("sample_seed", -1)),
+                                "normalized_action": np.asarray(
+                                    model_diag.get("normalized_action", action_chunk), dtype=np.float32
+                                ),
+                                "flow_states": np.asarray(model_diag.get("flow_states", []), dtype=np.float32),
+                                "velocity_norms": np.asarray(
+                                    model_diag.get("velocity_norms", []), dtype=np.float32
+                                ),
+                            }
+                        )
                     pending_actions = build_pending_actions(
                         action_chunk,
                         start_timestamp=steps,
                         action_exec_horizon=action_exec_horizon,
                         ensembler=ensembler,
                     )
-                action = libero_policy_action_to_env(pending_actions.pop(0))
+                dataset_action = pending_actions.pop(0)
+                action = (
+                    gripper_controller.convert(dataset_action)
+                    if gripper_controller is not None
+                    else libero_policy_action_to_env(dataset_action)
+                )
                 recorded_actions.append(action.copy())
 
             obs, _, done, _ = env.step(action)
@@ -95,6 +138,16 @@ def record_episode(
     if action_path is not None:
         action_path.parent.mkdir(parents=True, exist_ok=True)
         np.save(action_path, np.asarray(recorded_actions, dtype=np.float32))
+    if diagnostic_path is not None:
+        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
+        payload = {}
+        if diagnostic_records:
+            for key in diagnostic_records[0]:
+                payload[key] = np.stack([record[key] for record in diagnostic_records])
+        payload["executed_actions"] = np.asarray(recorded_actions, dtype=np.float32)
+        payload["success"] = np.asarray(bool(done))
+        payload["steps"] = np.asarray(steps)
+        np.savez_compressed(diagnostic_path, **payload)
 
     return bool(done), steps, len(recorded_actions)
 
@@ -122,6 +175,15 @@ def parse_args(argv=None):
     parser.add_argument("--action-horizon", type=int, default=16)
     parser.add_argument("--action-exec-horizon", type=int, default=4)
     parser.add_argument("--use-action-ensemble", action="store_true")
+    parser.add_argument("--use-gripper-hysteresis", action="store_true")
+    parser.add_argument("--gripper-close-threshold", type=float, default=0.35)
+    parser.add_argument("--gripper-open-threshold", type=float, default=0.65)
+    parser.add_argument("--gripper-min-hold-steps", type=int, default=10)
+    parser.add_argument(
+        "--image-transform",
+        choices=("rotate180", "vertical", "horizontal", "none"),
+        default="rotate180",
+    )
     parser.add_argument("--num-wait-steps", type=int, default=5)
     parser.add_argument("--seed", type=int, default=0)
     parser.add_argument("--dtype", default="bfloat16")
@@ -139,6 +201,8 @@ def parse_args(argv=None):
     parser.add_argument("--fps", type=int, default=20)
     parser.add_argument("--camera", default="agentview_image", help="Observation key used for video frames.")
     parser.add_argument("--save-actions", action="store_true", help="Also save executed actions as .npy.")
+    parser.add_argument("--save-diagnostics", action="store_true", help="Save model inputs and flow trajectory.")
+    parser.add_argument("--flow-seed", type=int, default=None, help="Deterministic action-flow noise seed.")
     parser.add_argument("--random-policy", action="store_true")
     parser.add_argument("--no-progress", action="store_true", help="Disable rollout tqdm progress bar.")
     return parser.parse_args(argv)
@@ -183,6 +247,8 @@ def main(argv=None):
         lang_hidden_size=args.lang_hidden_size,
         gen_hidden_size=args.gen_hidden_size,
         act_hidden_size=args.act_hidden_size,
+        flow_seed=args.flow_seed,
+        record_flow_diagnostics=args.save_diagnostics,
     )
     if not args.random_policy and hasattr(policy.backend, "_get_model"):
         print("[libero_vis] loading checkpoint into GPU...", flush=True)
@@ -203,6 +269,9 @@ def main(argv=None):
         output_dir = Path(args.output_dir)
         video_path = output_dir / f"{stem}.mp4"
         action_path = output_dir / f"{stem}_actions.npy" if args.save_actions else None
+        diagnostic_path = output_dir / f"{stem}_diagnostics.npz" if args.save_diagnostics else None
+        if args.flow_seed is not None:
+            policy.reset_flow_seed(args.flow_seed + args.trial_id * 100_000 + task_id * 1_000)
         max_steps = _get_max_steps(suite_name) + args.num_wait_steps
         print(f"[libero_vis] recording {video_path.name} (up to {max_steps} steps)", flush=True)
         try:
@@ -216,9 +285,15 @@ def main(argv=None):
                 num_wait_steps=args.num_wait_steps,
                 video_path=video_path,
                 action_path=action_path,
+                diagnostic_path=diagnostic_path,
                 camera=args.camera,
                 fps=args.fps,
                 use_action_ensemble=args.use_action_ensemble,
+                use_gripper_hysteresis=args.use_gripper_hysteresis,
+                gripper_close_threshold=args.gripper_close_threshold,
+                gripper_open_threshold=args.gripper_open_threshold,
+                gripper_min_hold_steps=args.gripper_min_hold_steps,
+                image_transform=args.image_transform,
                 show_progress=not args.no_progress,
             )
         finally:
@@ -231,6 +306,8 @@ def main(argv=None):
         _print_path("video", video_path)
         if action_path is not None:
             _print_path("actions", action_path)
+        if diagnostic_path is not None:
+            _print_path("diagnostics", diagnostic_path)
 
 
 if __name__ == "__main__":
diff --git a/training/egowam_native/eval/policy.py b/training/egowam_native/eval/policy.py
index af30c8e..64bb4d6 100644
--- a/training/egowam_native/eval/policy.py
+++ b/training/egowam_native/eval/policy.py
@@ -87,6 +87,8 @@ class TorchEgoWAMNativeBackend:
         lang_hidden_size: int | None = 1024,
         gen_hidden_size: int | None = 1024,
         act_hidden_size: int = 1024,
+        flow_seed: int | None = None,
+        record_flow_diagnostics: bool = False,
         model_loader: Callable[[], object] | None = None,
         observation_packer: Callable[[dict], dict] | None = None,
         action_sampler: Callable[[object, dict], np.ndarray] | None = None,
@@ -111,6 +113,10 @@ class TorchEgoWAMNativeBackend:
         self.lang_hidden_size = _optional_positive_int(lang_hidden_size)
         self.gen_hidden_size = _optional_positive_int(gen_hidden_size)
         self.act_hidden_size = int(act_hidden_size)
+        self.flow_seed = None if flow_seed is None else int(flow_seed)
+        self.record_flow_diagnostics = bool(record_flow_diagnostics)
+        self._flow_sample_index = 0
+        self.last_diagnostics: dict = {}
         self._model_loader = model_loader
         self._observation_packer = observation_packer
         self._action_sampler = action_sampler
@@ -131,6 +137,14 @@ class TorchEgoWAMNativeBackend:
         action = self._sample_action(model, batch)
         return validate_action_chunk(action, action_horizon=self.action_horizon, action_dim=self.action_dim)
 
+    def reset_flow_seed(self, seed: int | None = None) -> None:
+        if seed is not None:
+            self.flow_seed = int(seed)
+        self._flow_sample_index = 0
+
+    def get_last_diagnostics(self) -> dict:
+        return self.last_diagnostics
+
     def _get_model(self):
         if self._model is None:
             self._model = self._model_loader() if self._model_loader is not None else self._load_model()
@@ -294,7 +308,26 @@ class TorchEgoWAMNativeBackend:
         if device.type == "cuda":
             torch.cuda.set_device(device)
         batch = _move_packed_batch_to_device(batch, device, non_blocking=False)
-        action_z = torch.randn(1, self.action_horizon, self.action_dim, device=device, dtype=dtype)
+        sample_seed = None
+        generator = None
+        flow_seed = getattr(self, "flow_seed", None)
+        sample_index = int(getattr(self, "_flow_sample_index", 0))
+        if flow_seed is not None:
+            sample_seed = int(flow_seed + sample_index)
+            generator = torch.Generator(device=device)
+            generator.manual_seed(sample_seed)
+        self._flow_sample_index = sample_index + 1
+        action_z = torch.randn(
+            1,
+            self.action_horizon,
+            self.action_dim,
+            device=device,
+            dtype=dtype,
+            generator=generator,
+        )
+        record_diagnostics = bool(getattr(self, "record_flow_diagnostics", False))
+        flow_states = [action_z[0].detach().float().cpu().numpy()] if record_diagnostics else None
+        velocity_norms = [] if record_diagnostics else None
         steps = max(1, self.num_inference_steps)
         timesteps, deltas = shifted_flow_inference_schedule(
             num_inference_steps=steps,
@@ -315,9 +348,24 @@ class TorchEgoWAMNativeBackend:
                 output = model(**batch, action_z=action_z, action_t=action_t)
                 if output.action_pred is None:
                     raise RuntimeError("Ego-WAM Native model did not return `action_pred` for action sampling.")
+                if velocity_norms is not None:
+                    velocity_norms.append(float(output.action_pred.detach().float().norm().cpu()))
                 action_z = action_z + delta * output.action_pred
-        action = action_z[0].detach().float().cpu().numpy().astype(np.float32)
-        return self._get_action_state_normalizer().denormalize_action(action)
+                if flow_states is not None:
+                    flow_states.append(action_z[0].detach().float().cpu().numpy())
+        normalized_action = action_z[0].detach().float().cpu().numpy().astype(np.float32)
+        action = self._get_action_state_normalizer().denormalize_action(normalized_action)
+        if record_diagnostics:
+            self.last_diagnostics = {
+                "sample_seed": sample_seed,
+                "normalized_action": normalized_action.copy(),
+                "denormalized_action": np.asarray(action, dtype=np.float32).copy(),
+                "flow_states": np.stack(flow_states).astype(np.float32),
+                "velocity_norms": np.asarray(velocity_norms, dtype=np.float32),
+                "timesteps": timesteps.detach().float().cpu().numpy().astype(np.float32),
+                "deltas": deltas.detach().float().cpu().numpy().astype(np.float32),
+            }
+        return action
 
 
 def _optional_positive_int(value):
@@ -523,6 +571,15 @@ class EgoWAMNativePolicy:
         action = self.backend.predict(observation)
         return validate_action_chunk(action, action_horizon=self.action_horizon, action_dim=self.action_dim)
 
+    def reset_flow_seed(self, seed: int | None = None) -> None:
+        reset = getattr(self.backend, "reset_flow_seed", None)
+        if reset is not None:
+            reset(seed)
+
+    def get_last_diagnostics(self) -> dict:
+        getter = getattr(self.backend, "get_last_diagnostics", None)
+        return getter() if getter is not None else {}
+
 
 def build_policy(
     *,
@@ -546,6 +603,8 @@ def build_policy(
     lang_hidden_size: int | None = 1024,
     gen_hidden_size: int | None = 1024,
     act_hidden_size: int = 1024,
+    flow_seed: int | None = None,
+    record_flow_diagnostics: bool = False,
 ) -> EgoWAMNativePolicy:
     if random_policy:
         backend: PolicyBackend = RandomPolicyBackend(action_horizon=action_horizon, action_dim=action_dim, seed=seed)
@@ -569,5 +628,7 @@ def build_policy(
             lang_hidden_size=lang_hidden_size,
             gen_hidden_size=gen_hidden_size,
             act_hidden_size=act_hidden_size,
+            flow_seed=flow_seed,
+            record_flow_diagnostics=record_flow_diagnostics,
         )
     return EgoWAMNativePolicy(backend=backend, action_horizon=action_horizon, action_dim=action_dim)
diff --git a/training/tests/test_egowam_native_contracts.py b/training/tests/test_egowam_native_contracts.py
index 2f67dd1..124cfa4 100644
--- a/training/tests/test_egowam_native_contracts.py
+++ b/training/tests/test_egowam_native_contracts.py
@@ -670,6 +670,27 @@ def test_libero_policy_action_to_env_maps_and_binarizes_gripper_without_mutating
     np.testing.assert_array_equal(converted[:, -1], np.asarray([-1.0, 1.0, -1.0], dtype=np.float32))
 
 
+def test_libero_gripper_hysteresis_separates_thresholds_and_holds_state():
+    from egowam_native.eval.libero_rollout import LiberoGripperHysteresis
+
+    controller = LiberoGripperHysteresis(
+        close_threshold=0.35,
+        open_threshold=0.65,
+        min_hold_steps=2,
+    )
+
+    def convert(value):
+        action = np.zeros(7, dtype=np.float32)
+        action[-1] = value
+        return controller.convert(action)[-1]
+
+    assert convert(0.5) == -1.0  # dead band retains initial open state
+    assert convert(0.2) == 1.0  # close immediately
+    assert convert(0.8) == 1.0  # opening blocked by minimum hold
+    assert convert(0.8) == 1.0
+    assert convert(0.8) == -1.0
+
+
 def test_libero_rollout_sends_converted_gripper_action_to_env():
     from egowam_native.eval.libero_rollout import run_episode
 

```


# 附录C：原始分析结果位置

```text

/root/wuqingman/RUN/7-0712_libero_4suite_gen_act_stable_100k/failure_analysis/

```


核心报告：

```text

/root/wuqingman/RUN/7-0712_libero_4suite_gen_act_stable_100k/failure_analysis/root_cause_report.md

```
