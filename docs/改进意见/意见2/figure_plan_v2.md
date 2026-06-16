# 后续图像绘制清单 (Figure Plan v2)

本阶段不要求绘制最终图，仅列清单。临时 Python 图已生成至 `experiments/figures/`，需用专业绘图工具重绘为期刊级图片。

## 图 1：CUDA Block/Target 映射结构图
- **目的**：说明 1 block/target 映射、block 内线程分工、单 kernel 全迭代
- **要求**：二维结构示意图，中文标注，标注 Grid(N)/Block(128)/thread 分工/shared memory/register LDLT
- **现有临时图**：`fig1_block_target_mapping.png`

## 图 2：Raw/Valid Throughput 对比
- **目的**：展示 CUDA-Mixed、cuRobo-Graph 随 N 变化的吞吐
- **要求**：横轴 N，纵轴 targets/s，同时显示 raw 和 valid，对 N=5000 cuRobo-Graph 异常点加注释
- **现有临时图**：`fig2_throughput_n_curve.png`

## 图 3：End-to-End 延迟对比
- **目的**：展示完整工程调用延迟（含 H2D/D2H）
- **要求**：横轴 N，纵轴 E2E time/ms，含 CUDA-Mixed 和 cuRobo-Graph
- **现有临时图**：`fig3_e2e_latency.png`

## 图 4：GPU Stream 时间对比
- **目的**：纯设备侧执行时间，性能主图
- **要求**：横轴 N，纵轴 GPU stream time/ms，与图 3 风格统一
- **现有临时图**：`fig4_gpu_stream_time.png`

## 图 5：误差分布对比
- **目的**：展示 CUDA-Mixed 与 cuRobo-Graph 求解质量差异
- **要求**：箱线图或 p50/p95 柱状图，位置误差/mm + 姿态误差/deg
- **现有临时图**：`fig8_error_distribution.png`（需改进）

## 图 6：消融实验图
- **目的**：展示 A5→A7 的吞吐和收敛率变化
- **要求**：双轴图，主图仅展示 A0/A5/A7 关键节点
- **现有临时图**：`fig6_ablation.png`

## 图 7：cuRobo 批量退化分析
- **目的**：展显 cuRobo-NoGraph 在特定 N 下的退化
- **要求**：横轴 N，纵轴 GPU stream time，标出退化点
- **现有临时图**：`fig5_curobo_degradation.png`

## 图 8：Nsight Compute 剖析对比
- **目的**：计算吞吐、Bank 冲突、寄存器等指标
- **可用表格为主，图为辅**
- **现有临时图**：`fig9_ncu_profiling.png`

注：已删除原"A7/A8 CUDA Graph 对比图"（原图 6），因为复核后该对比意义有限（差异 <1% 或受 GPU 状态影响）。
