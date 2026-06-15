%% 图9: FP32/FP64混合精度计算路径
% 图表类型: 精度转换流程图 (precision transition flow diagram)
% 设计思路:
%   - 左侧FP32蓝色区域: FK + Jacobian (90%运算量, 利用64x FP32吞吐)
%   - 右侧FP64虚线红框: H/g累加 + LDL^T (关键路径, 保证精度)
%   - 箭头标注精度转换点 FP32→FP64 和 FP64→FP32
%   - 底部说明Ada Lovelace 1:64吞吐比

clear; close all;

figure('Position', [50 50 1200 700], 'Color', 'w');
hold on; axis off; xlim([0 12]); ylim([0 7]);

% FP32区域
annotation('rectangle', [0.05 0.50 0.40 0.38], 'EdgeColor', [0 0.4 0.8], 'LineWidth', 2, 'FaceAlpha', 0.15);
annotation('textbox', [0.05 0.84 0.12 0.04], 'String', 'FP32 区域', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0 0.4 0.8], 'EdgeColor', 'none');

% FP64区域
annotation('rectangle', [0.50 0.50 0.45 0.38], 'EdgeColor', [0.8 0.2 0], 'LineWidth', 2, 'LineStyle', '--', 'FaceAlpha', 0.10);
annotation('textbox', [0.50 0.84 0.16 0.04], 'String', 'FP64 关键路径', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.8 0.2 0], 'EdgeColor', 'none');

% 模块 (使用简化辅助函数)
r = @(x,y,w,h,c,s) annotation('textbox', [x y w h], 'String', s, 'BackgroundColor', c, ...
    'FontSize', 9, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'EdgeColor', 'k', 'LineWidth', 1);

r(0.08, 0.70, 0.18, 0.12, [0.8 0.9 1], 'UR10 FK (链式乘法) [FP32]');
r(0.08, 0.56, 0.18, 0.12, [0.8 0.9 1], '数值Jacobian (12次FK) [FP32]');
r(0.52, 0.70, 0.20, 0.12, [1 0.85 0.75], 'H=J^TW^2J+lambda*I [FP64]');
r(0.52, 0.56, 0.20, 0.12, [1 0.85 0.75], 'g=J^TW^2e [FP64]');
r(0.76, 0.70, 0.18, 0.12, [1 0.7 0.6], 'LDL^T分解(86 ops) [FP64]');
r(0.76, 0.56, 0.18, 0.12, [0.8 0.9 1], '关节更新 q+Delta q [FP32]');

% 箭头
a = @(x1,y1,x2,y2) annotation('textarrow', [x1 x2], [y1 y2], 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5);
a(0.26, 0.76, 0.52, 0.76);
a(0.26, 0.62, 0.52, 0.62);
a(0.10, 0.56, 0.10, 0.68);
a(0.72, 0.76, 0.76, 0.76);
a(0.72, 0.62, 0.76, 0.62);

% 转换标注
annotation('textbox', [0.38, 0.74, 0.12, 0.05], 'String', 'FP32->FP64', 'FontSize', 8, 'FontWeight', 'bold', ...
    'Color', [0.8 0.2 0], 'EdgeColor', 'none', 'BackgroundColor', [1 0.95 0.8]);
annotation('textbox', [0.38, 0.60, 0.12, 0.05], 'String', 'FP64->FP32', 'FontSize', 8, 'FontWeight', 'bold', ...
    'Color', [0 0.4 0.8], 'EdgeColor', 'none', 'BackgroundColor', [0.85 0.95 1]);

% 底部说明
annotation('textbox', [0.10 0.20 0.80 0.12], 'String', ...
    'Ada Lovelace: FP64:FP32 = 1:64吞吐比 | FP32=计算密集型(90%运算) | FP64=精度敏感路径(LDL^T/阻尼/收敛) | 收敛率0.998+ Bank冲突-63% Kernel时间-72%', ...
    'FontSize', 9, 'HorizontalAlignment', 'center', 'EdgeColor', 'k', 'BackgroundColor', [0.95 0.95 0.95]);

title('图9  FP32/FP64混合精度计算路径', 'FontSize', 16, 'FontWeight', 'bold');
exportgraphics(gcf, 'fig9_mixed_precision.png', 'Resolution', 300);
disp('图9完成');
