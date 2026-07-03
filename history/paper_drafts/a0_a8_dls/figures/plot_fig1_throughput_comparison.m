%% 图1: CUDA B5与cuRobo吞吐量对比（分组柱状图，对数坐标）
% 数据源: main_comparison.csv
% 图表类型: 分组柱状图 (grouped bar chart)
% 关键信息: N=100/500/1000/5000 四个批量规模下 B5 vs cuRobo 的吞吐量对比
% 设计思路:
%   - Y轴对数坐标, 因为吞吐量跨越两个数量级 (3k~170k)
%   - B5蓝色柱 vs cuRobo橙色柱, 形成直观对比
%   - 柱顶标注加速比 (36.1x/10.0x/4.7x/1.09x)

clear; close all;

% 读取数据
main = readtable('/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/data/results/main_comparison/main_comparison.csv');
b5_rows = strcmp(main.Solver, 'CUDA B5');
cu_rows = strcmp(main.Solver, 'cuRobo');
N_vals = main.N(b5_rows);
b5_tp = main.Throughput_targets_per_s(b5_rows);
cu_tp = main.Throughput_targets_per_s(cu_rows);
speedups = {'36.1×', '10.0×', '4.7×', '1.09×'};

% 绘图
figure('Position', [100 100 900 600], 'Color', 'w');
b = bar(log10([b5_tp, cu_tp]), 'grouped');
b(1).FaceColor = [0.0 0.45 0.74];  % B5蓝色
b(2).FaceColor = [0.85 0.33 0.10]; % cuRobo橙色

set(gca, 'XTickLabel', {'N=100', 'N=500', 'N=1000', 'N=5000'}, 'FontSize', 12);
ylabel('吞吐量 (targets/s, 对数坐标)', 'FontSize', 12);
title('图1  CUDA B5与cuRobo批量IK吞吐量对比', 'FontSize', 14, 'FontWeight', 'bold');
legend([b(1) b(2)], {'CUDA B5 (混合精度)', 'cuRobo (PyTorch)'}, 'Location', 'northwest', 'FontSize', 11);
grid on; box on;

% 标注加速比
for i = 1:4
    text(i-0.15, log10(b5_tp(i))+0.08, speedups{i}, 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0 0.3 0.8]);
end

% Y轴对数刻度标签
yt = [1000 5000 10000 50000 100000 200000];
set(gca, 'YTick', log10(yt), 'YTickLabel', {'1,000', '5,000', '10,000', '50,000', '100,000', '200,000'});

exportgraphics(gcf, 'fig1_throughput_comparison.png', 'Resolution', 300);
disp('图1完成');
