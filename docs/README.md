# SAKF-IK Research Showcase

面向小矩阵批量逆运动学的 CUDA 单核函数融合求解 — 项目展示网站。

## 在线访问

部署到 GitHub Pages 后可通过以下地址访问：

```
https://<username>.github.io/cuda_ik_accelerateion/
```

## 本地预览

直接在浏览器中打开 `index.html` 即可：

```bash
# Linux
xdg-open index.html

# macOS
open index.html

# 或使用 Python HTTP Server
python3 -m http.server 8080
# 然后访问 http://localhost:8080
```

## 网站内容

| 区块 | 内容 |
|------|------|
| Hero | 论文标题、作者信息、关键指标 |
| Section 01 | Python / C++ / CUDA 三种计算方案性能对比 |
| Section 02 | CUDA 硬件架构基础（线程层次、内存层次） |
| Section 03 | 机械臂 GPU 加速研究现状（2023–2025） |
| Section 04 | cuRobo 详解 + cuBLAS/cuSOLVER 范式级错配分析 |
| Section 05 | **SAKF-IK 四层单核函数融合架构（核心）** |
| Section 06 | 实验结果深度分析（消融、对比、微架构） |
| Section 07 | 总结与参考文献 |

## 技术栈

- 纯 HTML/CSS/JS，零框架依赖
- KaTeX CDN 数学公式渲染
- Prism.js CDN 代码语法高亮
- Google Fonts（Inter、Noto Sans SC、Noto Serif SC、JetBrains Mono）
- Intersection Observer 滚动动画

## 文件结构

```
gh-pages-site/
├── index.html              # 主页面
├── README.md               # 本文件
└── assets/
    ├── base.css            # 样式表
    ├── runtime.js          # 交互脚本
    └── figures/            # 论文图片（20 张）
```
