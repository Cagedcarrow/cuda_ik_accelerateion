%% 图6: 三级收敛阈值 × 五种N值 吞吐矩阵分析
% 数据源: threshold_scan.csv
% 图表类型: 热力图 + 气泡图 并排 (heatmap + bubble chart side-by-side)
% 关键信息:
%   - 左图热力图: 行=阈值(宽松/中等/严格), 列=N(100/500/1000/5000/10000), 色深=吞吐量
%   - 右图气泡图: X=N, Y=阈值, 气泡大小=加速比, 气泡颜色=加速比
%   - 核心发现: Medium阈值下36.1x→1.09x, Strict阈值下B5仍保持优势(1.05x)

clear; close all;

ts = readtable('/mnt/linuxdata/cuda_ik_accelerateion/standard_robot_cuda_ik/data/results/threshold_scan/threshold_scan.csv');

threshold_keys = {'Loose', 'Medium', 'Strict'};
threshold_labels = {'宽松', '中等', '严格'};
N_list = [100, 500, 1000, 5000, 10000];
tp_mat = zeros(3, 5);
sp_mat = zeros(3, 5);

for ti = 1:3
    for ni = 1:5
        row = ts.N == N_list(ni) & contains(ts.Threshold, threshold_keys{ti});
        tp_mat(ti, ni) = ts.CUDA_B5_TP(row) / 1000;
        sp_mat(ti, ni) = ts.CUDA_B5_TP(row) / ts.cuRobo_TP(row);
    end
end

figure('Position', [100 100 1100 480], 'Color', 'w');

subplot(1,2,1);
imagesc(tp_mat);
colormap(gca, parula);
c = colorbar;
c.Label.String = '吞吐量 (x10^3 targets/s)';
set(gca, 'XTick', 1:5, 'XTickLabel', {'N=100','N=500','N=1000','N=5000','N=10000'}, 'FontSize', 10);
set(gca, 'YTick', 1:3, 'YTickLabel', threshold_labels, 'FontSize', 10);
title('CUDA B5 吞吐量热力图', 'FontSize', 12, 'FontWeight', 'bold');
for ti = 1:3
    for ni = 1:5
        txtcolor = 'w'; if tp_mat(ti,ni) > 140; txtcolor = 'k'; end
        text(ni, ti, sprintf('%.0fk', tp_mat(ti,ni)), 'FontSize', 9, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'Color', txtcolor);
    end
end

subplot(1,2,2);
[Ng, Tg] = meshgrid(N_list, 1:3);
scatter(Ng(:), Tg(:), sp_mat(:)*80, sp_mat(:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
colormap(gca, jet);
c2 = colorbar;
c2.Label.String = '加速比 B5/cuRobo';
set(gca, 'YTick', 1:3, 'YTickLabel', threshold_labels, 'FontSize', 10);
xlabel('批量规模 N', 'FontSize', 11);
title('加速比气泡图', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
for ti = 1:3
    for ni = 1:5
        text(N_list(ni)+180, ti+0.15, sprintf('%.1fx', sp_mat(ti,ni)), 'FontSize', 7, 'HorizontalAlignment', 'center');
    end
end

sgtitle('图6  三级收敛阈值吞吐矩阵分析', 'FontSize', 14, 'FontWeight', 'bold');
exportgraphics(gcf, 'fig6_threshold_heatmap.png', 'Resolution', 300);
disp('图6完成');
