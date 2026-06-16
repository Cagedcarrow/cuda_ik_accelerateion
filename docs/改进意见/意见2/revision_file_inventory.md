# Revision File Inventory (v2)

| 类别 | 路径 | 用途 |
|------|------|------|
| 论文 LaTeX 源文件 | `docs/latex论文/paper.tex` | 需修改 |
| 论文 PDF | `docs/latex论文/基于 CUDA 小矩阵加速的机械臂批量逆运动学求解.pdf` | 输出目标 |
| 论文 PDF 备份 | `docs/latex论文_backup_v2_20260615_130404/` | v2 修改前备份 |
| CUDA benchmark runner | `standard_robot_cuda_ik/src/cuda/cuda_benchmark_runner.cu` | 已修改（H2D/D2H/多阈值/error-log） |
| CUDA kernel source | `standard_robot_cuda_ik/src/cuda/cuda_ik_6dof.cu` | 参考（不修改） |
| cuRobo benchmark | `standard_robot_cuda_ik/benchmark/bench_curobo.py` | 已修改（CUDA Graph/GPU event） |
| Common benchmark | `standard_robot_cuda_ik/benchmark/common.py` | 已修改（新字段） |
| 主实验数据 | `standard_robot_cuda_ik/experiments/01_official_ur10_main/performance_summary.csv` | 复核/更新 |
| 消融数据 | `standard_robot_cuda_ik/experiments/04_ablation/ablation_full.csv` | 参考 |
| Jacobian 精度数据 | `standard_robot_cuda_ik/experiments/05_roofline/jacobian_precision_summary.csv` | 已验证 |
| cuRobo 验证脚本 | `standard_robot_cuda_ik/experiments/00_verification/verify_curobo_graph_n5000.py` | Step 3 |
| A7/A8 验证脚本 | `standard_robot_cuda_ik/experiments/00_verification/verify_a7_a8_graph.py` | Step 4 |
| 图片生成脚本 | `standard_robot_cuda_ik/experiments/plot_all_new_figures.py` | 参考（需更新） |
| 已有图片 | `standard_robot_cuda_ik/experiments/figures/` | 9 张 PNG |
| 模板 | `docs/系统工程与电子技术/参考论文/模板/20210316135031_500.docx` | 格式参考 |
