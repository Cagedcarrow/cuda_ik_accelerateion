%% 图11: 寄存器级6x6 LDL^T求解流程
% 图表类型: 算法流程图 (algorithm flowchart)
% 设计思路:
%   - 四个阶段: 分解(35 FMA) → 前代(15 FMA) → 对角缩放(6 DIV) → 回代(15 FMA)
%   - 每阶段框内写伪代码
%   - 关键特性标注: 0次sqrt, 98 regs, ~0.1us
%   - 底部汇总: 65 FMA + 21 DIV = 86标量运算

clear; close all;

figure('Position', [50 50 1000 750], 'Color', 'w');
hold on; axis off; xlim([0 10]); ylim([0 8]);

title('图11  寄存器级6x6 LDL^T求解流程 (86次标量运算)', 'FontSize', 16, 'FontWeight', 'bold');

% 四个阶段框
stages = {
    '阶段1: 分解 (35 FMA)', [0.05 0.58 0.40 0.30];
    '阶段2: 前代 Ly=b (15 FMA)', [0.52 0.58 0.40 0.30];
    '阶段3: 对角缩放 z=D^{-1}y (6 DIV)', [0.05 0.15 0.40 0.30];
    '阶段4: 回代 L^Tx=z (15 FMA)', [0.52 0.15 0.40 0.30];
};
colors_s = {[0.8 0.9 1], [0.8 1 0.8], [1 0.9 0.8], [1 0.85 0.85]};
for i = 1:4
    annotation('textbox', stages{i,2}, 'String', stages{i,1}, 'BackgroundColor', colors_s{i}, ...
        'FontSize', 11, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'EdgeColor', 'k', 'LineWidth', 1.5, 'FontWeight', 'bold');
end

% 阶段1伪代码
annotation('textbox', [0.08 0.60 0.34 0.25], 'String', 'for j=0:5 (逐列约化)', 'FontSize', 8, 'HorizontalAlignment', 'left', 'EdgeColor', 'none');
annotation('textbox', [0.08 0.50 0.34 0.25], 'String', '  D_j = H_jj - sum(L_jk^2 D_k)', 'FontSize', 8, 'HorizontalAlignment', 'left', 'EdgeColor', 'none');
annotation('textbox', [0.08 0.40 0.34 0.25], 'String', '  for i=j+1:5', 'FontSize', 8, 'HorizontalAlignment', 'left', 'EdgeColor', 'none');
annotation('textbox', [0.08 0.30 0.34 0.25], 'String', '    L_ij = (H_ij - sum) / D_j', 'FontSize', 8, 'HorizontalAlignment', 'left', 'EdgeColor', 'none');

% 阶段2伪代码
annotation('textbox', [0.55 0.60 0.34 0.25], 'String', 'for i=0:5', 'FontSize', 8, 'HorizontalAlignment', 'left', 'EdgeColor', 'none');
annotation('textbox', [0.55 0.52 0.34 0.25], 'String', '  y_i = b_i - sum(L_ij y_j)', 'FontSize', 8, 'HorizontalAlignment', 'left', 'EdgeColor', 'none');

% 阶段3伪代码
annotation('textbox', [0.08 0.17 0.34 0.25], 'String', 'for i=0:5', 'FontSize', 8, 'HorizontalAlignment', 'left', 'EdgeColor', 'none');
annotation('textbox', [0.08 0.09 0.34 0.25], 'String', '  z_i = y_i / D_i', 'FontSize', 8, 'HorizontalAlignment', 'left', 'EdgeColor', 'none');

% 阶段4伪代码
annotation('textbox', [0.55 0.17 0.34 0.25], 'String', 'for i=5:-1:0 (倒序回代)', 'FontSize', 8, 'HorizontalAlignment', 'left', 'EdgeColor', 'none');
annotation('textbox', [0.55 0.09 0.34 0.25], 'String', '  x_i = z_i - sum(L_ji x_j)', 'FontSize', 8, 'HorizontalAlignment', 'left', 'EdgeColor', 'none');

% 箭头
a = @(x1,y1,x2,y2) annotation('textarrow', [x1 x2], [y1 y2], 'Color', [0.2 0.2 0.2], 'LineWidth', 1.5);
a(0.25, 0.58, 0.25, 0.50);
a(0.52, 0.73, 0.52, 0.73);

% 底部总结
annotation('textbox', [0.20 0.002 0.60 0.06], 'String', ...
    '合计: 65 FMA + 21 DIV = 86标量运算 | 编译时 #pragma unroll 完全展开 | 零sqrt (vs Cholesky) | 零local memory spill', ...
    'FontSize', 10, 'HorizontalAlignment', 'center', 'EdgeColor', 'k', 'BackgroundColor', [0.9 0.95 1], 'LineWidth', 1.5);

% 关键特性
annotation('textbox', [0.85 0.82 0.12 0.14], 'String', {'0次 sqrt'; '98 regs'; '~0.1 us'}, ...
    'BackgroundColor', [0.9 1 0.9], 'FontSize', 9, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'EdgeColor', [0 0.6 0], 'LineWidth', 2);

exportgraphics(gcf, 'fig11_ldlt_flow.png', 'Resolution', 300);
disp('图11完成');
