# CUDA IK V3：面向 cuRobo 的GPU原生IK系统升级计划

---

# 0. 项目目标（重新定义“冲击 cuRobo”）

## 0.1 总目标

构建一个**GPU 原生 IK 求解框架 V3**，在以下三个维度上对 cuRobo 形成系统级竞争：

### 必须达成目标（Hard Targets）

* N ≤ 1000 时：

  * ✔ End-to-End latency ≤ cuRobo-Graph
* N ≤ 5000 时：

  * ✔ GPU throughput ≥ cuRobo-Graph（或 ≥ 80%）
* Strict accuracy：

  * ✔ ≥ cuRobo（position ≤ 1 mm级别）
* Success Rate：

  * ✔ ≥ cuRobo（≥ 95% Medium / ≥ 85% Strict）

---

## 0.2 核心战略变化（非常关键）

V1/V2：

> 优化 DLS 求解器

V3：

> 构建 GPU 原生 IK optimization engine（不是 DLS）

---

# 1. 系统总体架构（V3）

## 1.1 总体流水线

```
Input Targets (T)
        ↓
GPU Batch Scheduler
        ↓
Multi-Seed Initialization Engine
        ↓
LM / Gauss-Newton Core Solver
        ↓
Line Search + Trust Region Controller
        ↓
Convergence Manager
        ↓
Best Solution Selector (per target)
        ↓
Output Joint Angles
```

---

# 2. 核心升级模块设计

---

## 2.1 Module A：解析 Jacobian（已完成 V2 → V3强化）

### 目标

替换所有数值差分 Jacobian

---

### 实现要求

#### A1：解析 Jacobian Kernel

* 每个 joint 对应：

  * geometric Jacobian row computation
* 使用：

  * rotation axis + position cross product

---

### GPU设计

```
block = 1 target
thread = 6 joints × 6 DOF contributions
```

---

### 优化点

* shared memory缓存 link transforms
* register reuse link chain
* eliminate FK重复计算

---

## 2.2 Module B：LM（Levenberg–Marquardt）GPU实现（核心升级）

### 目标

替代 DLS：

```
(JᵀJ + λI)Δq = -Jᵀe
```

---

### 改进点

#### B1：自适应阻尼 λ

```
λ = λ0 * (loss_t / loss_{t-1})
```

---

#### B2：GPU parallel solver

* 6×6 system per block
* register-level Cholesky / LDLT
* warp-synchronous reduction

---

#### B3：step acceptance

```
if loss_new < loss_old:
    accept step
    λ = λ / 2
else:
    reject step
    λ = λ * 2
```

---

## 2.3 Module C：Multi-Seed IK Engine（决定是否能赢 cuRobo）

### 目标

解决 DLS 单解局限

---

### 结构

```
N targets
 × K seeds (K=8~32)
 = N × K parallel IK instances
```

---

### Seed策略

* zero seed
* jitter seed (±0.1 rad noise)
* joint limit biased seed
* random reachable seed
* last-frame warm start

---

### GPU布局

```
grid.x = N
grid.y = K
block = IK solver
```

---

### reduction

per target:

```
argmin_k loss(q_k)
```

---

## 2.4 Module D：Trust Region / Line Search（关键提升Strict SR）

### 目标

提升：

* Strict SR from ~82% → 95%+

---

### 方法

#### D1：trust region constraint

```
||Δq|| ≤ Δ_max
```

---

#### D2：backtracking line search

```
q_new = q + α Δq
α ∈ {1, 0.5, 0.25, ...}
```

---

#### D3：singularity protection

```
det(JJᵀ) < ε → increase λ
```

---

## 2.5 Module E：GPU Batch Scheduler（决定是否超过 cuRobo Graph）

### 目标

减少 kernel launch variance

---

### 策略

* dynamic batch packing
* fixed-size tile execution
* warp-cooperative IK groups

---

### CUDA Graph integration

```
Graph capture per batch size
Graph replay per iteration
```

---

# 3. CUDA Kernel 架构（V3核心）

---

## 3.1 Kernel 结构

### single kernel per iteration（保持）

但内容变为：

```
Kernel V3:
  1. FK (analytic)
  2. Jacobian (analytic)
  3. error vector
  4. LM matrix build
  5. 6×6 solve
  6. line search
  7. state update
```

---

## 3.2 Memory layout升级

### 原 V2

* PaddedMat6×8

### V3

#### per target layout:

```
struct IKState {
    float q[6];
    float J[36];
    float e[6];
    float H[36];
    float g[6];
}
```

---

### 优化目标

* fully register-resident (preferred)
* shared fallback only for multi-seed

---

# 4. 性能优化目标（关键）

---

## 4.1 kernel launch

* V3目标：

```
1 kernel per iteration (unchanged)
BUT
- fewer iterations
- fewer failures
```

---

## 4.2 iteration reduction

| Method              | iterations |
| ------------------- | ---------- |
| V1 DLS              | 14–16      |
| V2 analytic DLS     | 10–14      |
| V3 LM + line search | 5–8        |

---

## 4.3 expected speedup breakdown

| component              | speedup         |
| ---------------------- | --------------- |
| analytic Jacobian      | 5×              |
| LM convergence         | 2×              |
| multi-seed parallelism | 4–16× effective |
| memory optimization    | 1.2–1.5×        |

---

# 5. Benchmark体系（必须对标 cuRobo）

---

## 5.1 对比对象

* cuRobo-Graph (ON/OFF)
* cuRobo baseline
* V2 CUDA DLS (baseline)
* V3 LM system

---

## 5.2 指标体系

### Performance

* end-to-end latency
* GPU stream time
* valid throughput

---

### Quality

* position error mean / p95
* rotation error mean / p95
* success rate (Medium / Strict)

---

### Stability

* std deviation of latency
* worst-case latency
* convergence variance

---

## 5.3 critical benchmark sets

```
N = 100
N = 500
N = 1000
N = 2000
N = 5000
N = 10000
```

---

## 5.4 special stress tests

* near singular configurations
* joint limit boundary cases
* unreachable targets
* random workspace corners

---

# 6. 预期对 cuRobo 的“真实竞争结构”

---

## 6.1 expected outcome model

| region          | expected result                 |
| --------------- | ------------------------------- |
| N=100           | V3 wins or ties                 |
| N=500           | close                           |
| N=1000          | competitive                     |
| N=5000          | cuRobo slightly better or equal |
| strict accuracy | cuRobo better                   |

---

## 6.2 论文级正确结论（目标）

不是：

> “我们全面超过 cuRobo”

而是：

> “我们在 GPU IK pipeline 的某些结构维度（低延迟 + 可预测性 + kernel simplicity）上与 cuRobo 达到同级竞争，并在极小批量场景中具有优势”

---

# 7. 代码工程结构（建议）

```
cuda_ik_v3/
├── core/
│   ├── fk_analytic.cu
│   ├── jacobian_analytic.cu
│   ├── lm_solver.cu
│   ├── line_search.cu
│   └── ik_kernel_v3.cu
│
├── multi_seed/
│   ├── seed_generator.cu
│   ├── seed_scheduler.cu
│
├── batch/
│   ├── batch_manager.cu
│   ├── graph_executor.cu
│
├── benchmark/
│   ├── curobo_adapter.py
│   ├── metrics.py
│   ├── eval_pipeline.py
│
└── docs/
    ├── experiments.md
    ├── results.md
```

---

# 8. milestone计划

---

## M1（已完成 V2基础）

* analytic Jacobian ✔

---

## M2（核心升级）

* LM solver GPU版
* line search
* trust region

---

## M3（决定胜负）

* multi-seed GPU parallel

---

## M4（对标 cuRobo）

* full benchmark
* CUDA Graph integration
* Nsight profiling

---

# 9. 成败判据（非常关键）

---

## 成功标准（必须全部满足 ≥1）

### ✔ 性能胜利（弱胜）

* N=100 latency < cuRobo-Graph

---

### ✔ 系统胜利（强胜）

* N≤1000 end-to-end competitive
* iteration count significantly lower
* kernel launch complexity lower

---

### ✔ 论文胜利（最现实）

* clear advantage in:

  * low batch IK
  * deterministic latency
  * simple GPU execution model

---

# 10. 最重要结论（写给论文用）

> V3 的目标不是替代 cuRobo，而是在 GPU IK 计算空间中提供一种“低延迟、结构简洁、可预测性强”的替代范式，并通过 LM + multi-seed GPU 并行，将 IK 求解从单路径优化扩展为多候选并行搜索问题。

---

# END
