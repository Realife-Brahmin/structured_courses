% hw02_qB2.m
% EE582 HW02 - Section B.2: Inductor SVGT equations (Trapz & BE)
% Author: Aryan Ritwajeet Jha
% Date: October 2025

setup_plot_theme(); % Consistent plot style

% Circuit parameters
V_DC = 10;      % Source voltage [V]
L = 0.02;       % Inductance [H]
R = 10;         % Resistance [Ohm]
h_steps_ms = [0.1, 0.8]; % Time steps [ms]
T_horizon_ms = 10;
T_horizon = T_horizon_ms * 1e-3; % Simulation time [s]

% --- RL Trapezoidal and BE update (corrected) ---
results = struct();
for h_idx = 1:length(h_steps_ms)
    h_ms = h_steps_ms(h_idx);
    h = h_ms * 1e-3;
    t = 0:h:T_horizon;
    N = length(t);
    % Initial conditions
    i_trapz = zeros(1, N); vL_trapz = zeros(1, N);
    i_BE = zeros(1, N); vL_BE = zeros(1, N);
    % Trapezoidal parameters
    R_L_trapz = 2*L/h;
    e_h_trapz = 0;
    for n = 1:N
        % Trapezoidal: compute i(t) then vL(t)
        i_trapz(n) = (V_DC + e_h_trapz) / (R + R_L_trapz);
        vL_trapz(n) = R_L_trapz * i_trapz(n) - e_h_trapz;
        % Update history term for next step
        e_h_trapz = R_L_trapz * i_trapz(n) + vL_trapz(n);
    end
    % Backward Euler parameters
    R_L_BE = L/h;
    e_h_BE = 0;
    for n = 1:N
        % BE: i(t) = (V_DC + e_h_BE) / (R + R_L_BE)
        i_BE(n) = (V_DC + e_h_BE) / (R + R_L_BE);
        vL_BE(n) = R_L_BE * i_BE(n) - e_h_BE;
        % Update history term for next step
        e_h_BE = R_L_BE * i_BE(n);
    end
    results(h_idx).h_ms = h_ms;
    results(h_idx).t = t;
    results(h_idx).i_trapz = i_trapz;
    results(h_idx).vL_trapz = vL_trapz;
    results(h_idx).i_BE = i_BE;
    results(h_idx).vL_BE = vL_BE;
    results(h_idx).R_L_trapz = R_L_trapz;
    results(h_idx).R_L_trapz = R_L_trapz;
    results(h_idx).R_L_trapz = R_L_trapz;
    results(h_idx).R_L_BE = R_L_BE;

% ...existing code...

% --- End of for-loop ---

end

%% Plot Results


% Color and marker config
col_trapz_dark = [0.85, 0.33, 0.10];   % Dark orange
col_trapz_light = [1.0, 0.65, 0.30];   % Light orange
col_be_dark = [0.00, 0.45, 0.74];      % Dark blue
col_be_light = [0.40, 0.70, 1.00];     % Light blue
marker_trapz = 'o';
marker_be = 's';
ms_small = 3.5; ms_large = 7;
lw_thick = 2.5; lw_thin = 1.5;

figure('Name', 'Inductor SVGT: Trapz vs BE (Both h)', 'Color', 'w', 'Position', [100 100 700 900]);
sgtitle({'\textbf{MATLAB Simulation: RL Circuit}', ...
    sprintf('$V_{\\mathrm{DC}} = %.0f$ V, $L = %.3f$ H, $R = %.1f~\\Omega$, $T = %.1f$ ms', V_DC, L, R, T_horizon*1e3)}, ...
    'Interpreter', 'latex', 'FontSize', 13);

% --- Voltage subplot (top) ---
subplot(2,1,1); hold on;
% Trapz h=0.1 ms
plot(results(1).t*1e3, results(1).vL_trapz, '-', 'Color', col_trapz_dark, 'LineWidth', lw_thick, ...
    'Marker', marker_trapz, 'MarkerIndices', 1:length(results(1).t), 'MarkerSize', ms_small, 'DisplayName', 'Trapz, $h=0.1$ ms', 'MarkerFaceColor', col_trapz_dark, 'MarkerEdgeColor', col_trapz_dark);
% Trapz h=0.8 ms
plot(results(2).t*1e3, results(2).vL_trapz, '-', 'Color', col_trapz_light, 'LineWidth', lw_thin, ...
    'Marker', marker_trapz, 'MarkerIndices', 1:length(results(2).t), 'MarkerSize', ms_large, 'DisplayName', 'Trapz, $h=0.8$ ms', 'MarkerFaceColor', col_trapz_light, 'MarkerEdgeColor', col_trapz_light);
% BE h=0.1 ms
plot(results(1).t*1e3, results(1).vL_BE, '-', 'Color', col_be_dark, 'LineWidth', lw_thick, ...
    'Marker', marker_be, 'MarkerIndices', 1:length(results(1).t), 'MarkerSize', ms_small, 'DisplayName', 'BE, $h=0.1$ ms', 'MarkerFaceColor', col_be_dark, 'MarkerEdgeColor', col_be_dark);
% BE h=0.8 ms
plot(results(2).t*1e3, results(2).vL_BE, '-', 'Color', col_be_light, 'LineWidth', lw_thin, ...
    'Marker', marker_be, 'MarkerIndices', 1:length(results(2).t), 'MarkerSize', ms_large, 'DisplayName', 'BE, $h=0.8$ ms', 'MarkerFaceColor', col_be_light, 'MarkerEdgeColor', col_be_light);
grid on;
xlabel('Time [ms]', 'Interpreter', 'latex');
ylabel('Inductor Voltage $v_L(t)$ [V]', 'Interpreter', 'latex');
title('Inductor Voltage', 'Interpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');
% --- Current subplot (bottom) ---
subplot(2,1,2); hold on;
% Trapz h=0.1 ms
plot(results(1).t*1e3, results(1).i_trapz, '-', 'Color', col_trapz_dark, 'LineWidth', lw_thick, ...
    'Marker', marker_trapz, 'MarkerIndices', 1:length(results(1).t), 'MarkerSize', ms_small, 'DisplayName', 'Trapz, $h=0.1$ ms', 'MarkerFaceColor', col_trapz_dark, 'MarkerEdgeColor', col_trapz_dark);
% Trapz h=0.8 ms
plot(results(2).t*1e3, results(2).i_trapz, '-', 'Color', col_trapz_light, 'LineWidth', lw_thin, ...
    'Marker', marker_trapz, 'MarkerIndices', 1:length(results(2).t), 'MarkerSize', ms_large, 'DisplayName', 'Trapz, $h=0.8$ ms', 'MarkerFaceColor', col_trapz_light, 'MarkerEdgeColor', col_trapz_light);
% BE h=0.1 ms
plot(results(1).t*1e3, results(1).i_BE, '-', 'Color', col_be_dark, 'LineWidth', lw_thick, ...
    'Marker', marker_be, 'MarkerIndices', 1:length(results(1).t), 'MarkerSize', ms_small, 'DisplayName', 'BE, $h=0.1$ ms', 'MarkerFaceColor', col_be_dark, 'MarkerEdgeColor', col_be_dark);
% BE h=0.8 ms
plot(results(2).t*1e3, results(2).i_BE, '-', 'Color', col_be_light, 'LineWidth', lw_thin, ...
    'Marker', marker_be, 'MarkerIndices', 1:length(results(2).t), 'MarkerSize', ms_large, 'DisplayName', 'BE, $h=0.8$ ms', 'MarkerFaceColor', col_be_light, 'MarkerEdgeColor', col_be_light);
grid on;
xlabel('Time [ms]', 'Interpreter', 'latex');
ylabel('Inductor Current $i_L(t)$ [A]', 'Interpreter', 'latex');
title('Inductor Current', 'Interpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');

% Save figure
figuresFolder = "../tex_Hw02/figures/";
if ~exist(figuresFolder, 'dir'), mkdir(figuresFolder); end
saveas(gcf, fullfile(figuresFolder, 'hw02_qB-2_matlab-cktB-svtg-comparison-trapz-vs-be-100mus-vs-800mus.png'));

% --- Print Tables (first 6 steps for each method/step) ---
for h_idx = 1:length(h_steps_ms)
    fprintf('\nTrapezoidal: R_L = %.6f Ohm\n', results(h_idx).R_L_trapz);
    fprintf('=== Trapezoidal Table (h = %.1f ms) ===\n', h_steps_ms(h_idx));
    fprintf('%-8s %-12s %-12s\n', 't [ms]', 'v_L(t) [V]', 'i(t) [A]');
    fprintf('%s\n', repmat('-',1,36));
    for n = 1:min(6, length(results(h_idx).t))
        fprintf('%-8.3f %-12.6f %-12.6f\n', results(h_idx).t(n)*1e3, results(h_idx).vL_trapz(n), results(h_idx).i_trapz(n));
    end
    fprintf('\nBackward Euler: R_L = %.6f Ohm\n', results(h_idx).R_L_BE);
    fprintf('=== Backward Euler Table (h = %.1f ms) ===\n', h_steps_ms(h_idx));
    fprintf('%-8s %-12s %-12s\n', 't [ms]', 'v_L(t) [V]', 'i(t) [A]');
    fprintf('%s\n', repmat('-',1,36));
    for n = 1:min(6, length(results(h_idx).t))
        fprintf('%-8.3f %-12.6f %-12.6f\n', results(h_idx).t(n)*1e3, results(h_idx).vL_BE(n), results(h_idx).i_BE(n));
    end
end
