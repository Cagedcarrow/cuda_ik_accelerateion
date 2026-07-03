%% 图7: 批量IK CUDA加速系统总体架构
% 图表类型: 系统流程图 (system flowchart / block diagram)
% 无数据源 - 纯架构图, 使用 annotation 手绘
% 设计思路:
%   - 从左到右: URDF模型 → FK/IK参数解析 → 常量内存导出 → Kernel Launch → DLS迭代循环 → 收敛统计输出
%   - 中下部展开Block内部: 10个DLS阶段顺序执行, 收敛判断分支(否→继续迭代, 是→输出)
%   - 右侧标注内存层次

clear; close all;

figure('Position', [50 50 1400 800], 'Color', 'w');
hold on; axis off; xlim([0 14]); ylim([0 8]);

boxColor = [0.9 0.95 1];
boxColor2 = [0.95 0.9 1];
arrowColor = [0.3 0.3 0.3];
memColor = [1 0.95 0.8];
kernelColor = [0.8 1 0.8];

% 第一行: 输入阶段
annotation('textbox', [0.02 0.72 0.12 0.06], 'String', 'URDF模型', 'BackgroundColor', boxColor, ...
    'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', 'EdgeColor', 'k');
annotation('textarrow', [0.15 0.19], [0.75 0.75], 'Color', arrowColor, 'LineWidth', 1.5);

annotation('textbox', [0.20 0.72 0.14 0.06], 'String', {'FK/IK参数'; '解析'}, 'BackgroundColor', boxColor, ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'EdgeColor', 'k');
annotation('textarrow', [0.35 0.39], [0.75 0.75], 'Color', arrowColor, 'LineWidth', 1.5);

annotation('textbox', [0.40 0.72 0.16 0.06], 'String', {'常量内存导出'; '(7数组 1,384B)'}, ...
    'BackgroundColor', memColor, 'HorizontalAlignment', 'center', 'FontSize', 10, 'EdgeColor', 'k');
annotation('textarrow', [0.57 0.61], [0.75 0.75], 'Color', arrowColor, 'LineWidth', 1.5);

% Kernel Launch 核心
annotation('textbox', [0.62 0.72 0.18 0.06], 'String', {'CUDA Kernel Launch'; '<<<N, 128>>>'}, ...
    'BackgroundColor', kernelColor, 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', 'EdgeColor', 'k', 'LineWidth', 2);
annotation('textarrow', [0.81 0.85], [0.75 0.75], 'Color', arrowColor, 'LineWidth', 1.5);

% Block内部结构（虚线大框）
annotation('rectangle', [0.05 0.15 0.90 0.52], 'EdgeColor', [0 0.3 0.7], 'LineWidth', 2.5, 'LineStyle', '--');

% DLS 迭代循环 (上下两排5个阶段)
phases = {{'FK'; '(thread 0)'}, {'位姿误差'; '(thread 0)'}, {'收敛?'; '(thread 0)'}, ...
    {'Jacobian'; '(thread 0-5)'}, {'自适应λ'; '(thread 0)'}, ...
    {'H矩阵'; '(thread 0-35)'}, {'g向量'; '(thread 0-5)'}, ...
    {'LDL^T求解'; '(thread 0)'}, {'步长钳位'; '(thread 0)'}, {'关节更新'; '(thread 0-5)'}};

for i = 1:10
    x = 0.08 + (i-1)*0.085;
    if i <= 5; y = 0.55; else; y = 0.35; end
    if i == 1 || i == 4 || i == 6 || i == 8 || i == 10
        bg = boxColor;
    else
        bg = [0.95 0.95 0.95];
    end
    annotation('textbox', [x y 0.075 0.10], 'String', phases{i}, 'BackgroundColor', bg, ...
        'HorizontalAlignment', 'center', 'FontSize', 7.5, 'EdgeColor', 'k', 'VerticalAlignment', 'middle');
    if i < 10
        annotation('textarrow', [x+0.08 x+0.11], [y+0.05 y+0.05], 'Color', arrowColor);
    end
end

% 收敛判断分支（向下箭头）
annotation('textarrow', [0.335 0.335], [0.55 0.30], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
annotation('textbox', [0.25 0.25 0.08 0.04], 'String', '否 继续', 'Color', 'r', 'EdgeColor', 'none', 'FontSize', 7);

% 回到FK的循环箭头
annotation('textarrow', [0.07 0.07], [0.48 0.58], 'Color', [0 0.6 0], 'LineWidth', 1.2);

% 输出
annotation('textbox', [0.86 0.72 0.12 0.06], 'String', {'收敛统计'; '输出'}, 'BackgroundColor', boxColor2, ...
    'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', 'EdgeColor', 'k');

% 右侧内存层次
annotation('textbox', [0.82 0.42 0.16 0.06], 'String', {'共享内存'; 'PaddedMat6x8'}, 'BackgroundColor', [1 0.9 0.8], ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'EdgeColor', 'k');
annotation('textbox', [0.82 0.24 0.16 0.06], 'String', {'寄存器'; 'LDL^T求解器'}, 'BackgroundColor', [0.8 0.9 1], ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'EdgeColor', 'k');

title('图7  批量IK CUDA加速系统总体架构', 'FontSize', 16, 'FontWeight', 'bold');
exportgraphics(gcf, 'fig7_system_architecture.png', 'Resolution', 300);
disp('图7完成');
