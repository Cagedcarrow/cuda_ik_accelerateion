%% 图8: Block内线程分工与内存层次数据流
% 图表类型: 层级映射图 (hierarchical mapping diagram)
% 设计思路:
%   - 顶层: Grid(N,1,1) N个Block并行
%   - 中层: 单个Block(128 threads)内部9个阶段的thread映射框
%   - 右侧: 共享内存(PaddedMat6x8)、寄存器(LDL^T)、常量内存 三级内存层次
%   - 红色标注: H矩阵跨越Warp0+Warp1的关键警示

clear; close all;

figure('Position', [50 50 1300 850], 'Color', 'w');
hold on; axis off; xlim([0 13]); ylim([0 8.5]);

% Grid层
annotation('textbox', [0.35 0.90 0.30 0.05], 'String', 'Grid: (N, 1, 1) -- N个目标并行', ...
    'BackgroundColor', [0.8 0.9 1], 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'EdgeColor', 'k', 'LineWidth', 2);

% N个Block示意
for i = 1:5
    x = 0.05 + (i-1)*0.18;
    annotation('rectangle', [x 0.78 0.16 0.10], 'EdgeColor', [0 0.3 0.7], 'LineWidth', 1.5);
    annotation('textbox', [x 0.80 0.16 0.06], 'String', sprintf('Block %d', i), ...
        'FontSize', 8, 'HorizontalAlignment', 'center', 'EdgeColor', 'none');
end
annotation('textbox', [0.90 0.80 0.08 0.06], 'String', '...', 'FontSize', 14, 'EdgeColor', 'none');

% 单个Block展开（虚线框）
annotation('rectangle', [0.05 0.05 0.90 0.70], 'EdgeColor', [0 0.5 0], 'LineWidth', 2.5, 'LineStyle', '--');
annotation('textbox', [0.05 0.72 0.35 0.04], 'String', 'Block (128 threads) 内部阶段式分工', ...
    'FontSize', 11, 'FontWeight', 'bold', 'Color', [0 0.5 0], 'EdgeColor', 'none');

% 各阶段线程映射
phases = {
    'FK计算', 'threadIdx.x == 0', 0.07, 0.60, [0.9 0.95 1];
    '位姿误差', 'threadIdx.x == 0', 0.07, 0.50, [0.95 0.95 0.95];
    '收敛判定', 'threadIdx.x == 0', 0.07, 0.40, [0.9 0.95 1];
    '数值Jacobian', 'threadIdx.x < 6', 0.32, 0.60, [0.8 1 0.8];
    'H矩阵构造', 'threadIdx.x < 36', 0.32, 0.50, [1 0.85 0.75];
    'g向量构造', 'threadIdx.x < 6', 0.32, 0.40, [0.8 1 0.8];
    'LDL^T求解', 'threadIdx.x == 0', 0.57, 0.60, [0.8 0.85 1];
    '步长钳位', 'threadIdx.x == 0', 0.57, 0.50, [0.95 0.95 0.95];
    '关节更新', 'threadIdx.x < 6', 0.57, 0.40, [0.8 1 0.8];
};

for i = 1:size(phases,1)
    x = phases{i,3}; y = phases{i,4};
    annotation('textbox', [x y 0.22 0.07], 'String', [phases{i,1} '  (' phases{i,2} ')'], ...
        'BackgroundColor', phases{i,5}, 'FontSize', 8.5, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'EdgeColor', 'k');
end

% 共享内存区域
annotation('textbox', [0.82 0.62 0.15 0.05], 'String', '共享内存', 'BackgroundColor', [1 0.9 0.7], ...
    'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'EdgeColor', 'k');
annotation('textbox', [0.82 0.50 0.15 0.10], 'String', {'PaddedMat6x8'; 'J(48 FP64)'; 'H(48 FP64)'; '1,616 bytes'}, ...
    'BackgroundColor', [1 0.95 0.85], 'FontSize', 8, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'EdgeColor', 'k');

% 寄存器区域
annotation('textbox', [0.82 0.35 0.15 0.05], 'String', '寄存器', 'BackgroundColor', [0.7 0.85 1], ...
    'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'EdgeColor', 'k');
annotation('textbox', [0.82 0.22 0.15 0.11], 'String', {'LDL^T 6x6'; '86标量运算'; '98 regs/thread'; '零spill'}, ...
    'BackgroundColor', [0.8 0.9 1], 'FontSize', 8, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'EdgeColor', 'k');

% 常量内存区域
annotation('textbox', [0.82 0.10 0.15 0.05], 'String', '常量内存', 'BackgroundColor', [0.85 1 0.85], ...
    'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'EdgeColor', 'k');

% 关键警示
annotation('textbox', [0.30 0.68 0.10 0.04], 'String', 'Warp0+Warp1跨越!', 'Color', 'r', ...
    'FontSize', 8, 'FontWeight', 'bold', 'EdgeColor', 'r', 'BackgroundColor', [1 0.9 0.9]);

title('图8  Block内线程分工与内存层次数据流', 'FontSize', 16, 'FontWeight', 'bold');
exportgraphics(gcf, 'fig8_thread_mapping.png', 'Resolution', 300);
disp('图8完成');
