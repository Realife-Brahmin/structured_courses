% hw02_qB3.m
% Section B.3: RL Circuit Smackdown - PSCAD, Trapz, BE (2 h each)
% Combines results from PSCAD (100us, 800us), Trapz (0.1ms, 0.8ms), BE (0.1ms, 0.8ms)

setup_plot_theme();

% --- Load PSCAD Data ---
processedDataFolder = "../processedData/Hw02/";
pscad_100us = parse_pscad_output_qB(fullfile(processedDataFolder, 'hw02_qB_100mus'), 'hw02_qB_100mus');
pscad_800us = parse_pscad_output_qB(fullfile(processedDataFolder, 'hw02_qB_800mus'), 'hw02_qB_800mus');

% --- Load Trapz/BE Data (from hw02_qB2.m) ---
% Recompute to ensure up-to-date
V_DC = 10; L = 0.02; R = 10;
h_steps_ms = [0.1, 0.8];
T_horizon_ms = 10;
T_horizon = T_horizon_ms * 1e-3;
results = struct();
for h_idx = 1:length(h_steps_ms)
    h_ms = h_steps_ms(h_idx);
    h = h_ms * 1e-3;
    t = 0:h:T_horizon;
    N = length(t);
    i_trapz = zeros(1, N); vL_trapz = zeros(1, N);
    i_BE = zeros(1, N); vL_BE = zeros(1, N);
    R_L_trapz = 2*L/h;
    e_h_trapz = 0;
    for n = 1:N
        i_trapz(n) = (V_DC + e_h_trapz) / (R + R_L_trapz);
        vL_trapz(n) = R_L_trapz * i_trapz(n) - e_h_trapz;
        e_h_trapz = R_L_trapz * i_trapz(n) + vL_trapz(n);
    end
    R_L_BE = L/h;
    e_h_BE = 0;
    for n = 1:N
        i_BE(n) = (V_DC + e_h_BE) / (R + R_L_BE);
        vL_BE(n) = R_L_BE * i_BE(n) - e_h_BE;
        e_h_BE = R_L_BE * i_BE(n);
    end
    results(h_idx).h_ms = h_ms;
    results(h_idx).t = t;
    results(h_idx).i_trapz = i_trapz;
    results(h_idx).vL_trapz = vL_trapz;
    results(h_idx).i_BE = i_BE;
    results(h_idx).vL_BE = vL_BE;
end

%% --- Plotting ---
figure('Name', 'RL Circuit Solution Comparison: PSCAD vs Trapz vs BE', 'Color', 'w', 'Position', [100 100 800 1000]);
sgtitle({'MATLAB (Trapz, BE) vs PSCAD: RL Circuit', ...
    '$V_{\mathrm{DC}} = 10$ V, $L = 0.020$ H, $R = 10.0~\Omega$, $T = 10.0$ ms'}, ...
    'Interpreter', 'latex', 'FontSize', 14);

% Color/marker config
col_pscad_blue = [0.05, 0.28, 0.63]; % PSCAD blue
col_pscad_gold = [0.80, 0.65, 0.20]; % PSCAD gold
col_trapz_dark = [0.85, 0.33, 0.10];
col_trapz_light = [1.0, 0.65, 0.30];
col_be_dark = [0.00, 0.45, 0.74];
col_be_light = [0.40, 0.70, 1.00];

ms_pscad_tri = 9; ms_pscad_x = 10; ms_trapz = 5; ms_be = 5; ms_trapz_sq = 8; ms_be_sq = 8;
lw_pscad = 2.2; lw_pscad_x = 2.8; lw_trapz = 2.2; lw_be = 2.2; lw_thin = 1.2;


% --- Inductor Current i(t) ---
subplot(2,1,2); hold on;
% PSCAD 100us: cross, blue edge, yellow (cross has no fill, so use blue edge, yellow marker)
plot(pscad_100us.t*1e3, pscad_100us.IL, '-', 'Color', col_pscad_blue, 'LineWidth', lw_pscad, 'HandleVisibility','off');
scatter(pscad_100us.t*1e3, pscad_100us.IL, ms_pscad_x^2, 'Marker', 'x', 'MarkerEdgeColor', col_pscad_blue, 'LineWidth', lw_pscad_x, 'DisplayName', 'PSCAD $h=0.1$ ms');
% PSCAD 800us: triangle, yellow edge, blue fill
plot(pscad_800us.t*1e3, pscad_800us.IL, '-', 'Color', col_pscad_gold, 'LineWidth', lw_pscad, 'HandleVisibility','off');
scatter(pscad_800us.t*1e3, pscad_800us.IL, ms_pscad_tri^2, 'Marker', '^', 'MarkerEdgeColor', col_pscad_gold, 'MarkerFaceColor', col_pscad_blue, 'LineWidth', lw_pscad, 'DisplayName', 'PSCAD $h=0.8$ ms');
% Trapz h=0.1 ms
plot(results(1).t*1e3, results(1).i_trapz, 'o-', 'Color', col_trapz_dark, 'LineWidth', lw_trapz, 'MarkerSize', ms_trapz, 'MarkerFaceColor', col_trapz_light, 'DisplayName', 'Trapz $h=0.1$ ms');
% Trapz h=0.8 ms
plot(results(2).t*1e3, results(2).i_trapz, 's-', 'Color', col_trapz_light, 'LineWidth', lw_thin, 'MarkerSize', ms_trapz_sq, 'MarkerFaceColor', col_trapz_dark, 'DisplayName', 'Trapz $h=0.8$ ms');
% BE h=0.1 ms
plot(results(1).t*1e3, results(1).i_BE, 'o-', 'Color', col_be_dark, 'LineWidth', lw_be, 'MarkerSize', ms_be, 'MarkerFaceColor', col_be_light, 'DisplayName', 'BE $h=0.1$ ms');
% BE h=0.8 ms
plot(results(2).t*1e3, results(2).i_BE, 's-', 'Color', col_be_light, 'LineWidth', lw_thin, 'MarkerSize', ms_be_sq, 'MarkerFaceColor', col_be_dark, 'DisplayName', 'BE $h=0.8$ ms');
grid on; xlim padded; ylim padded;
xlabel('Time [ms]', 'Interpreter', 'latex');
ylabel('Inductor Current $i(t)$ [A]', 'Interpreter', 'latex');
title('Inductor Current', 'Interpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');
ax2 = gca; ax2.XMinorGrid = 'on'; ax2.YMinorGrid = 'on';


% --- Inductor Voltage v_L(t) ---
subplot(2,1,1); hold on;
% PSCAD 100us: cross, blue edge, yellow (cross has no fill, so use blue edge, yellow marker)
plot(pscad_100us.t*1e3, pscad_100us.vL, '-', 'Color', col_pscad_blue, 'LineWidth', lw_pscad, 'HandleVisibility','off');
scatter(pscad_100us.t*1e3, pscad_100us.vL, ms_pscad_x^2, 'Marker', 'x', 'MarkerEdgeColor', col_pscad_blue, 'LineWidth', lw_pscad_x, 'DisplayName', 'PSCAD $h=0.1$ ms');
% PSCAD 800us: triangle, yellow edge, blue fill
plot(pscad_800us.t*1e3, pscad_800us.vL, '-', 'Color', col_pscad_gold, 'LineWidth', lw_pscad, 'HandleVisibility','off');
scatter(pscad_800us.t*1e3, pscad_800us.vL, ms_pscad_tri^2, 'Marker', '^', 'MarkerEdgeColor', col_pscad_gold, 'MarkerFaceColor', col_pscad_blue, 'LineWidth', lw_pscad, 'DisplayName', 'PSCAD $h=0.8$ ms');
% Trapz h=0.1 ms
plot(results(1).t*1e3, results(1).vL_trapz, 'o-', 'Color', col_trapz_dark, 'LineWidth', lw_trapz, 'MarkerSize', ms_trapz, 'MarkerFaceColor', col_trapz_light, 'DisplayName', 'Trapz $h=0.1$ ms');
% Trapz h=0.8 ms
plot(results(2).t*1e3, results(2).vL_trapz, 's-', 'Color', col_trapz_light, 'LineWidth', lw_thin, 'MarkerSize', ms_trapz_sq, 'MarkerFaceColor', col_trapz_dark, 'DisplayName', 'Trapz $h=0.8$ ms');
% BE h=0.1 ms
plot(results(1).t*1e3, results(1).vL_BE, 'o-', 'Color', col_be_dark, 'LineWidth', lw_be, 'MarkerSize', ms_be, 'MarkerFaceColor', col_be_light, 'DisplayName', 'BE $h=0.1$ ms');
% BE h=0.8 ms
plot(results(2).t*1e3, results(2).vL_BE, 's-', 'Color', col_be_light, 'LineWidth', lw_thin, 'MarkerSize', ms_be_sq, 'MarkerFaceColor', col_be_dark, 'DisplayName', 'BE $h=0.8$ ms');
grid on; xlim padded; ylim padded;
xlabel('Time [ms]', 'Interpreter', 'latex');
ylabel('Inductor Voltage $v_L(t)$ [V]', 'Interpreter', 'latex');
title('Inductor Voltage', 'Interpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');
ax1 = gca; ax1.XMinorGrid = 'on'; ax1.YMinorGrid = 'on';

% --- Save Figure ---
figuresFolder = "../tex_Hw02/figures/";
if ~exist(figuresFolder, 'dir'), mkdir(figuresFolder); end
saveas(gcf, fullfile(figuresFolder, 'hw02_qB-3-pscad-vs-matlab-plots-for-vt-and-it-for-100mus-and-800mus.png'));
