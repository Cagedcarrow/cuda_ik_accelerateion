%% 图10: PaddedMat6x8共享内存Bank冲突降低原理
% 图表类型: 对比示意图 (side-by-side comparison diagram)
% 设计思路:
%   - 左侧: stride=6自然布局, 标注Bank索引(0-31), 红色标注冲突路径
%   - 右侧: stride=8 PaddedMat6x8, 偶数行绿色(Bank 0-15), 奇数行蓝色(Bank 16-31)
%   - 填充列灰色PAD标注
%   - 核心数学: gcd(12,32)=4 → 冲突; gcd(16,32)=16 → 两组Bank集不重叠

clear; close all;

figure('Position', [50 50 1400 650], 'Color', 'w');

% 左侧: stride=6 自然布局
subplot(1,2,1);
hold on;
title('stride=6 自然布局 (Bank冲突)', 'FontSize', 13, 'FontWeight', 'bold', 'Color', 'r');
for r = 1:6
    for c = 1:6
        bank = mod((r-1)*12 + (c-1)*2, 32);
        x = c; y = 7-r;
        if bank < 16
            color = [0.7 0.85 1];
        else
            color = [1 0.7 0.7];
        end
        rectangle('Position', [x-0.4 y-0.4 0.8 0.8], 'FaceColor', color, 'EdgeColor', 'k');
        text(x, y, sprintf('B%d', bank), 'FontSize', 8, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
end
xlim([0.2 6.8]); ylim([0.2 6.8]);
set(gca, 'XTick', 1:6, 'YTick', 1:6, 'YTickLabel', flip({'行1','行2','行3','行4','行5','行6'}), 'FontSize', 10);
xlabel('列', 'FontSize', 11);
grid on;
annotation('textbox', [0.08 0.35 0.3 0.06], 'String', 'gcd(12,32)=4  2-3路Bank冲突', ...
    'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold', 'EdgeColor', 'r', 'BackgroundColor', [1 0.9 0.9]);

% 右侧: stride=8 填充布局
subplot(1,2,2);
hold on;
title('stride=8 PaddedMat6x8 (Bank冲突降低)', 'FontSize', 13, 'FontWeight', 'bold', 'Color', [0 0.5 0]);
for r = 1:6
    for c = 1:8
        bank = mod((r-1)*16 + (c-1)*2, 32);
        x = c; y = 7-r;
        if c <= 6
            if mod(r,2) == 1
                color = [0.7 1 0.7];
            else
                color = [0.7 0.7 1];
            end
            rectangle('Position', [x-0.4 y-0.4 0.8 0.8], 'FaceColor', color, 'EdgeColor', 'k');
            text(x, y, sprintf('B%d', bank), 'FontSize', 8, 'HorizontalAlignment', 'center');
        else
            rectangle('Position', [x-0.4 y-0.4 0.8 0.8], 'FaceColor', [0.9 0.9 0.9], 'EdgeColor', 'k', 'LineStyle', ':');
            text(x, y, 'PAD', 'FontSize', 7, 'HorizontalAlignment', 'center', 'Color', [0.5 0.5 0.5]);
        end
    end
end
xlim([0.2 8.8]); ylim([0.2 6.8]);
set(gca, 'XTick', 1:8, 'YTick', 1:6, 'YTickLabel', flip({'行1','行2','行3','行4','行5','行6'}), 'FontSize', 10);
xlabel('列 (含2列填充)', 'FontSize', 11);
grid on;
annotation('textbox', [0.58 0.35 0.32 0.06], 'String', '偶数行Bank 0-15  奇数行Bank 16-31  两组不重叠', ...
    'Color', [0 0.5 0], 'FontSize', 10, 'FontWeight', 'bold', 'EdgeColor', [0 0.5 0], 'BackgroundColor', [0.9 1 0.9]);

sgtitle('图10  PaddedMat6x8共享内存Bank冲突降低原理', 'FontSize', 16, 'FontWeight', 'bold');
exportgraphics(gcf, 'fig10_bank_conflict.png', 'Resolution', 300);
disp('图10完成');
