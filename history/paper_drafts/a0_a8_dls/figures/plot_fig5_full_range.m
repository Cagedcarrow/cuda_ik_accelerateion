%% 图5: 全量程批量扩展性 N=100→10000（双线对比图）
% 数据源: full_range_comparison.csv
% 图表类型: 双线对比图 + 置信带 (dual-line comparison with confidence band)
% 关键信息:
%   - B5: 蓝实线圆点 + 148k-174k浅蓝带(±8%)
%   - cuRobo正常点: 绿三角虚线
%   - cuRobo退化点: 红色倒三角 + 红色误差棒
%   - 退化N值标注: 4000/7000/9000/10000

clear; close all;

fr = readtable('/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/data/results/full_range/full_range_comparison.csv');

N_data = [100 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000];
b5_tp = fr.CUDA_B5_TP_targets_per_s(1:12);
cu_tp = fr.cuRobo_TP_targets_per_s(1:12);
cu_status = fr.cuRobo_Status(1:12);

figure('Position', [100 100 1100 600], 'Color', 'w');
hold on;

fill([N_data fliplr(N_data)], [148000*ones(1,12) 174000*ones(1,12)], ...
    [0.0 0.45 0.74], 'FaceAlpha', 0.08, 'EdgeColor', 'none');
p1 = plot(N_data, b5_tp, 'b-o', 'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', 'b');

normal_idx = strcmp(cu_status, 'normal');
deg_idx = strcmp(cu_status, 'DEGRADED');
p2 = plot(N_data(normal_idx), cu_tp(normal_idx), '^--', 'Color', [0 0.6 0.3], 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0 0.6 0.3]);
p3 = plot(N_data(deg_idx), cu_tp(deg_idx), 'v', 'Color', [0.85 0.33 0.10], 'LineWidth', 2, 'MarkerSize', 12, 'MarkerFaceColor', 'r');

for i = 1:length(N_data)
    if deg_idx(i)
        text(N_data(i), cu_tp(i)-15000, '退化', 'FontSize', 9, 'Color', 'r', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end
end

set(gca, 'XTick', N_data, 'FontSize', 10);
xlabel('批量规模 N', 'FontSize', 12);
ylabel('吞吐量 (targets/s)', 'FontSize', 12);
title('图5  全量程批量扩展性 N=100→10000（Medium 10mm/5°）', 'FontSize', 14, 'FontWeight', 'bold');
legend([p1 p2 p3], {'CUDA B5 (148k-174k ±8%)', 'cuRobo 正常模式 (~32ms)', 'cuRobo 退化模式 (~230ms)'}, ...
    'Location', 'southeast', 'FontSize', 10);
grid on; box on;
ylim([0 280000]);

exportgraphics(gcf, 'fig5_full_range.png', 'Resolution', 300);
disp('图5完成');
