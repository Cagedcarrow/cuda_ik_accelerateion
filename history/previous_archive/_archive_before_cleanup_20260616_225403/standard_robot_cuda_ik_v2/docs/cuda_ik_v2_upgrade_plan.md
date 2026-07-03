# CUDA IK V2 升级方案：解析 Jacobian 技术验证

## 执行日期
2026-06-15

## 验证结果

### 核心数据（Python 原型，200 目标 UR10）

| 指标 | V1 数值 Jacobian | V2 解析 Jacobian | 结论 |
|------|-----------------|-----------------|------|
| **FP32 Jacobian 精度** | 中位误差 4.2×10⁻² | 中位误差 **9.6×10⁻⁸** | **436 000× 提升** |
| Strict SR (5mm/1°) | 79.5% | **82.0%** | 略有改善 |
| Medium SR (10mm/5°) | 82.5% | 85.0% | 改善 |
| 求解速度 (200 目标) | 8.5s | **1.6s** | **5.3× 加速** |
| FK 调用/迭代 | 12 | 1 | −91.7% |

### 关键发现

1. **FP32 精度革命性提升**：解析 Jacobian 不需要有限差分（ε），消除了减法抵消误差。FP32 解析 Jacobian 的中位误差仅 9.6×10⁻⁸——比数值 FP32 的 4.2×10⁻² 精确了 43.6 万倍。这意味着 FP32 混合精度路径不再受 Jacobian 精度限制。

2. **Strict SR 仅边际改善**：解析 Jacobian 的 Strict 收敛率从 79.5% → 82.0%（+2.5 个百分点）。提升有限的原因不在 Jacobian 精度，而在 DLS 算法本身的收敛极限——接近真解时梯度步长方向主要由 Hessian 决定，Jacobian 的精度改进收益递减。要根本性提升 Strict SR，需要更强的算法（如 L-BFGS、LM Trust-Region）。

3. **速度提升 5.3×**：主要来自 Jacobian 计算从 12 FK → 1 FK。

4. **解析 Jacobian 与数值 Jacobian 定义不同**：几何 Jacobian 的旋转分量是角速度，数值 Jacobian 的旋转分量是 log(R) 的导数。两者在 Frobenius 范数下有约 130% 的差异（因为定义不同），但在 DLS 迭代中均有效——这是预期行为。

### 对 V2 的影响评估

- ✅ 解析 Jacobian 的 CUDA 实现可行且预期收益大（5×+ 加速）
- ⚠️ 单独解析 Jacobian 不足以大幅提升 Strict SR
- ⚠️ 要真正冲击 cuRobo，需要结合更强的优化算法（LM Trust-Region 或多 seed 策略）

### 下一步：CUDA 实现
- `cuda_utilities.cuh`：新增 `forward_kinematics_with_frames()` —— 在 FK 过程中保存 p_i, z_i
- `cuda_ik_6dof.cu`：替换 Jacobian 计算段，用交叉积替代中心差分
- 共享内存增量：288 bytes（6×(3+3) FP64 → 288 bytes），总量 1616+288=1904 < 48KB
