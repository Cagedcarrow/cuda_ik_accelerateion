%% 图4: Nsight Compute剖析 B4 vs B5 平行坐标图
% 数据源: ncu_summary.csv
% 图表类型: 平行坐标图 (parallel coordinates plot)
% 关键信息:
%   - 8个NCU指标归一化对比: Compute/DRAM/Regs/Occ/Bank/L1/Spill/Duration
%   - 蓝线=B4(FP64全精度), 红线=B5(混合精度)
%   - 关键差异: Bank冲突 3522→1295 (-63%), Kernel时间 2920→827us (-72%)
%   - 共同特征: DRAM极低(1-2%), L1命中率极高(98-99%), Spill=0

clear; close all;

ncu = readtable('/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/data/profiling/ncu_summary.csv');

idx_b4 = contains(ncu.Config, 'B4') & ncu.N == 100;
idx_b5 = contains(ncu.Config, 'B5') & ncu.N == 100;

metrics = [ncu.ComputeThroughput_pct(idx_b4), ncu.ComputeThroughput_pct(idx_b5);
           ncu.DRAMThroughput_pct(idx_b4), ncu.DRAMThroughput_pct(idx_b5);
           ncu.RegistersPerThread(idx_b4), ncu.RegistersPerThread(idx_b5);
           ncu.Occupancy_pct(idx_b4), ncu.Occupancy_pct(idx_b5);
           ncu.BankConflicts(idx_b4), ncu.BankConflicts(idx_b5);
           ncu.L1HitRate_pct(idx_b4), ncu.L1HitRate_pct(idx_b5);
           ncu.KernelDuration_us(idx_b4), ncu.KernelDuration_us(idx_b5);
           0, 0];

metrics_norm = metrics;
max_vals = max(metrics, [], 2);
max_vals(max_vals==0) = 1;
for i = 1:size(metrics,1)
    metrics_norm(i,:) = metrics(i,:) / max_vals(i);
end

labels = {'计算吞吐率', 'DRAM吞吐率', '寄存器/线程', '占用率', 'Bank冲突', 'L1命中率', 'Kernel时间', '局部内存溢出'};

figure('Position', [100 100 1000 550], 'Color', 'w');
colors_p = {[0.0 0.45 0.74], [0.85 0.33 0.10]};
for i = 1:2
    plot(1:8, metrics_norm(:,i), 'o-', 'Color', colors_p{i}, 'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', colors_p{i});
    hold on;
end

set(gca, 'XTick', 1:8, 'XTickLabel', labels, 'FontSize', 10);
ylabel('归一化指标值', 'FontSize', 12);
title('图4  Nsight Compute剖析：B4(FP64) vs B5(混合精度) N=100', 'FontSize', 14, 'FontWeight', 'bold');
legend({'B4 (FP64全精度)', 'B5 (混合精度)'}, 'Location', 'northeast', 'FontSize', 11);
grid on; box on;

for i = 1:8
    text(i-0.15, metrics_norm(i,1)+0.04, sprintf('%.0f', metrics(i,1)), 'FontSize', 7, 'Color', colors_p{1});
    text(i+0.15, metrics_norm(i,2)-0.06, sprintf('%.0f', metrics(i,2)), 'FontSize', 7, 'Color', colors_p{2});
end

exportgraphics(gcf, 'fig4_ncu_parallel.png', 'Resolution', 300);
disp('图4完成');
