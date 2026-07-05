# LaTeX 工程说明

本目录提供论文源文件的最小可编译工程。

## 文件

- `paper.tex`：论文主 TeX 文件。
- `.latexmkrc`：latexmk 配置，使用 XeLaTeX 编译。
- `figures/`：论文正文通过 `\includegraphics` 引用的 PDF 图件。
- `CHANGELOG.md`：标题、章节和图表重构记录。

## 编译

需要安装 XeLaTeX 和 latexmk。在本目录运行：

```bash
latexmk -xelatex -interaction=nonstopmode paper.tex
```

当前工程使用 PDF 图件，不需要 Inkscape，也不需要 `-shell-escape`。
