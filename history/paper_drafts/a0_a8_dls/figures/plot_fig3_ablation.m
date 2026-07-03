%% 图3: 消融实验 B0→B3→B5（双Y轴: 柱状吞吐+折线收敛率）
% 数据源: ablation_medium.csv
% 图表类型: 分组柱状图 + 折线图 (grouped bar + line overlay)
% 关键信息:
%   - 柱状: B0(灰)/B3(绿)/B5(蓝) 吞吐量, 三组N(100/500/5000)
%   - 折线(右Y轴): 收敛率 (B0崩塌→B3/B5恢复100%)
%   - 柱顶标注提升百分比 (+577%/+530%/+428% 和 +120%/+149%/+149%)

clear; close all;

abl = readtable('/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/data/results/ablation/ablation_medium.csv');

levels = {'B0', 'B3', 'B5'};
N_vals = [100, 500, 5000];
colors = {[0.7 0.7 0.7], [0.0 0.6 0.3], [0.0 0.45 0.74]};

figure('Position', [100 100 1000 650], 'Color', 'w');
hold on;

tp_data = zeros(3, 3);
cr_data = zeros(3, 3);
for li = 1:3
    for ni = 1:3
        row = strcmp(abl.Level, levels{li}) & abl.N == N_vals(ni);
        tp_data(li, ni) = abl.Throughput_targets_per_s(row);
        cr_data(li, ni) = abl.ConvRate(row);
    end
end

bw = 0.2;
for li = 1:3
    x_pos = (1:3) + (li-2)*bw;
    bar(x_pos, tp_data(li,:)/1000, bw*0.9, 'FaceColor', colors{li}, 'EdgeColor', 'k', 'LineWidth', 0.5);
end

yyaxis right;
for li = 1:3
    plot(1:3, cr_data(li,:)*100, 'o-', 'Color', colors{li}, 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', colors{li});
end
ylabel('收敛率 (%)', 'FontSize', 12);
ylim([0 105]);
set(gca, 'YColor', 'k');

yyaxis left;
set(gca, 'XTick', 1:3, 'XTickLabel', {'N=100', 'N=500', 'N=5000'}, 'FontSize', 12);
ylabel('吞吐量 (×10^3 targets/s)', 'FontSize', 12);

pct_labels = {'+577%', '+530%', '+428%'; '+120%', '+149%', '+149%'};
for ni = 1:3
    text(ni, tp_data(2,ni)/1000+10, pct_labels{1,ni}, 'FontSize', 8, 'Color', colors{2}, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    text(ni, tp_data(3,ni)/1000+8, pct_labels{2,ni}, 'FontSize', 8, 'Color', colors{3}, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end

title('图3  消融实验：B0→B3→B5逐级优化效果', 'FontSize', 14, 'FontWeight', 'bold');
legend({'B0 (FP64基线)', 'B3 (+自适应阻尼)', 'B5 (+混合精度)'}, 'Location', 'northwest', 'FontSize', 11);
grid on; box on;

exportgraphics(gcf, 'fig3_ablation.png', 'Resolution', 300);
disp('图3完成');
