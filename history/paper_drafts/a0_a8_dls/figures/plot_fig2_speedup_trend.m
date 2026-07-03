%% 图2: B5/cuRobo加速比与吞吐趋势（双Y轴折线图）
% 数据源: main_comparison.csv
% 图表类型: 双Y轴折线+散点图 (dual-Y line+scatter)
% 关键信息:
%   - 左Y轴: 加速比 (36.1x→1.09x) 蓝色实线圆点
%   - 右Y轴: B5(蓝方块)和cuRobo(橙三角)的吞吐量趋势
%   - X轴: N=100/500/1000/5000

clear; close all;

main = readtable('/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/data/results/main_comparison/main_comparison.csv');
b5_rows = strcmp(main.Solver, 'CUDA B5');
cu_rows = strcmp(main.Solver, 'cuRobo');
N_vals = main.N(b5_rows);
b5_tp = main.Throughput_targets_per_s(b5_rows);
cu_tp = main.Throughput_targets_per_s(cu_rows);
speedup_vals = b5_tp ./ cu_tp;

figure('Position', [100 100 900 600], 'Color', 'w');
yyaxis left;
p1 = plot(1:4, speedup_vals, 'b-o', 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', 'b');
ylabel('吞吐比值 (B5/cuRobo)', 'FontSize', 12, 'Color', 'b');
set(gca, 'YColor', 'b');
ylim([0 40]);

yyaxis right;
p2 = plot(1:4, b5_tp/1000, 's-', 'Color', [0 0.45 0.74], 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', [0 0.45 0.74]);
hold on;
p3 = plot(1:4, cu_tp/1000, '^--', 'Color', [0.85 0.33 0.10], 'LineWidth', 2.5, 'MarkerSize', 10);
ylabel('吞吐量 (×10^3 targets/s)', 'FontSize', 12);

set(gca, 'XTick', 1:4, 'XTickLabel', {'N=100', 'N=500', 'N=1000', 'N=5000'}, 'FontSize', 12);
title('图2  加速比与吞吐量趋势', 'FontSize', 14, 'FontWeight', 'bold');
legend([p1 p2 p3], {'加速比 B5/cuRobo', 'CUDA B5 吞吐量', 'cuRobo 吞吐量'}, 'Location', 'northeast', 'FontSize', 10);
grid on; box on;

for i = 1:4
    text(i, speedup_vals(i)+1.5, sprintf('%.1f×', speedup_vals(i)), 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end

exportgraphics(gcf, 'fig2_speedup_trend.png', 'Resolution', 300);
disp('图2完成');
