# 终稿修订后审稿自检

## Nature polish 风格润色结果

- 摘要已改为“针对……问题，提出……方法。该方法首先……然后……最终……。实验结果表明……”结构。
- 中文摘要不含阿拉伯数字，按中文主体估算约 216 字，符合 200--250 字要求。
- 删除或降级了“完全来源于”“质量不可用”“天花板”“决定性”“灾难性”等偏绝对或口语化表达。
- cuRobo-K1 的表述改为“失败尾部较重”，同时承认其成功样本误差很低。
- 结论补充了不处理碰撞、不保证连续控制轨迹、不面向亚毫米级精密装配任务的边界。

## paper-review 七维自检

1. Novelty & Significance: Minor revision pass.
   论文贡献限定为固定六自由度机械臂、中小批量目标、无碰撞 IK 前端的结构感知 CUDA 实现，不再宣称全面优于通用求解器。

2. Methodology & Technical Soundness: Minor revision pass.
   统一评价协议、cuRobo-K1/K16、阈值定义、graph capture 计时边界、外部 FK 复核和特殊目标生成规则均已说明。

3. Results & Validation: Minor revision pass.
   表 6、阈值扫描、轨迹连续性和近奇异/近限位实验均保留数据支撑；有效吞吐量分析已承认 cuRobo-Graph-K1 在 throughput × Strict SR 指标上更高。

4. Reproducibility: Minor revision pass.
   随机种子、目标生成方式、物理关节编号与代码 0-based index 映射已补充；图表由 CSV 脚本重绘。

5. Related Work & Citations: Pass.
   参考文献共 25 条，当前检查显示全部被正文引用，无缺失引用。

6. Clarity & Organization: Minor revision pass.
   摘要、引言、实验分析和结论已按工程期刊语气收敛；核心图改为彩色并做了可读性检查。

7. Ethics & Limitations: Pass.
   局限性已明确：不含碰撞约束、不保证连续控制轨迹、不替代高精度通用运动生成框架。

## 仍需人工关注

- 图内保留了 OPT4C、cuRobo、K1/K16、Strict 等必要缩写。若投稿编辑严格要求图中文字全部中文化，可在排版阶段进一步替换为中文描述。
- 文题长度来自既定修订计划，若期刊严格执行中文题名不超过 20 个汉字，可能需要人工决定是否压缩题名。
