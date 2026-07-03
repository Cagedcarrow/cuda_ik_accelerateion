# OPT4C LaTeX Rewrite Compile Report

更新时间：2026-06-16 23:22 CST

## 1. 本轮修改

- 已备份旧稿：`paper_before_opt4c_rewrite_20260616_231937.tex`
- 已重写主稿：`paper.tex`
- 已重新编译：`paper.pdf`

本轮采用表格优先版本，不依赖外部图片文件；旧稿中的 `\includegraphics` 引用已移除。

## 2. 数据来源

最新论文数字只来自以下文件：

- `/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/PROJECT.md`
- `/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/data/results/latest/cuda_opt4c_static.csv`
- `/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/data/results/latest/cuda_vs_curobo_summary.csv`
- `/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/data/results/latest/latest_run_status.csv`

## 3. 关键写入数据

| N | CUDA throughput/s | CUDA Strict SR | CUDA pos_p95_all mm | cuRobo throughput/s | cuRobo Strict SR | cuRobo pos_p95_all mm |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 15539.2 | 0.960 | 4.385 | 10508.7 | 0.870 | 74.324 |
| 500 | 16344.0 | 0.954 | 4.338 | 41815.6 | 0.836 | 115.637 |
| 1000 | 17817.9 | 0.954 | 4.563 | 64928.2 | 0.840 | 98.920 |
| 5000 | 18490.4 | 0.954 | 4.563 | 137148.3 | 0.844 | 75.047 |

## 4. 编译命令

```bash
cd /mnt/linuxdata/cuda_ik_accelerateion/docs/latex论文_backup_v2_20260615_130404
latexmk -xelatex -interaction=nonstopmode paper.tex
```

## 5. 验证结果

| 检查项 | 结果 |
|---|---|
| `paper.pdf` 是否生成 | pass |
| PDF 页数 | 5 |
| PDF 创建时间 | 2026-06-16 23:22 CST |
| `paper.log` fatal/error 检查 | pass |
| 旧关键词检查 | pass |
| 外部图片依赖 | none |

旧关键词检查命令：

```bash
rg -n "128k|157k|A7|A8|CUDA-Mixed|standard_robot_cuda_ik_v4_cuda|includegraphics" paper.tex
```

结果为空。

## 6. 编译备注

LaTeX 编译成功，无 fatal error。日志中仍存在少量字体 warning、underfull/overfull box warning，属于排版质量问题，不影响 PDF 生成。若后续准备投稿版本，建议进一步手工压缩宽表、长路径和英文标题换行。
