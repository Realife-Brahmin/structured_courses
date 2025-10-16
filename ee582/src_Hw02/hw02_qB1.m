% hw02_qB1.m
% Plot PSCAD simulation results for Circuit B (RL circuit)
% Compares two time-step sizes: 100µs and 800µs

% Dynamically generate the folder path based on the script's location
currentScriptFolder = fileparts(mfilename('fullpath'));
cd(currentScriptFolder);
addpath(currentScriptFolder);

%% ===================== Setup Plot Theme =====================
setup_plot_theme();

% Superior color palette
color_100us = [0.85, 0.33, 0.10];  % Burnt Orange (100µs)
color_800us = [0.00, 0.45, 0.74];  % Deep Blue (800µs)

%% ===================== Circuit Parameters =====================
R = 10;             % Resistance [Ω]
L = 20e-3;          % Inductance [H] (20 mH)
V_DC = 10;          % DC voltage [V]
T_horizon_s = 10e-3; % Simulation duration [s] (10 ms)

processedDataFolder = "../processedData/Hw02/";

%% ===================== Parse PSCAD Data =====================
fprintf('\n=== Parsing PSCAD Data ===\n');

% Parse 100µs data
fprintf('\n--- 100µs Time Step ---\n');
pscad_folder_100us = fullfile(processedDataFolder, 'hw02_qB_100mus');
pscad_100us = parse_pscad_output_qB(pscad_folder_100us, 'hw02_qB_100mus');

% Parse 800µs data
fprintf('\n--- 800µs Time Step ---\n');
pscad_folder_800us = fullfile(processedDataFolder, 'hw02_qB_800mus');
pscad_800us = parse_pscad_output_qB(pscad_folder_800us, 'hw02_qB_800mus');

%% ===================== Plot Results =====================
fig1 = figure('Name', 'PSCAD Simulation: RL Circuit', 'Position', [100 100 1200 800], 'Color', 'w');

% Add main title with circuit parameters
sgtitle({'\textbf{PSCAD Simulation: RL Circuit}', ...
    sprintf('$R = %.0f$ $\\Omega$, $L = %.0f$ mH, $V_{\\mathrm{DC}} = %.0f$ V, $T = %.0f$ ms', ...
    R, L*1e3, V_DC, T_horizon_s*1e3)}, ...
    'Interpreter', 'latex', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

% Subplot 1: Inductor Voltage v_L(t)
subplot(2,1,1);
ax1 = gca;
plot(pscad_100us.t*1e3, pscad_100us.vL, 'o-', 'DisplayName', '$\Delta t = 100$ $\mu$s', ...
    'Color', color_100us, 'LineWidth', 3.5, 'MarkerSize', 6);
hold on;
plot(pscad_800us.t*1e3, pscad_800us.vL, 's-', 'DisplayName', '$\Delta t = 800$ $\mu$s', ...
    'Color', color_800us, 'LineWidth', 3.5, 'MarkerSize', 8);
grid on;
xlabel('Time [ms]', 'Interpreter', 'latex');
ylabel('Inductor Voltage $v_L(t)$ [V]', 'Interpreter', 'latex');
title('\textbf{Inductor Voltage}', 'Interpreter', 'latex', 'FontSize', 12, 'Color', 'k');
legend('Location', 'best', 'Interpreter', 'latex');
xlim([0 T_horizon_s*1e3]);
set(ax1, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'TickLabelInterpreter', 'latex');
% Explicit minor grid settings
ax1.XMinorGrid = 'on';
ax1.YMinorGrid = 'on';
ax1.MinorGridLineStyle = '-';
ax1.MinorGridAlpha = 0.15;
ax1.MinorGridColor = [0.5 0.5 0.5];

% Subplot 2: Inductor Current i(t)
subplot(2,1,2);
ax2 = gca;
plot(pscad_100us.t*1e3, pscad_100us.IL, 'o-', 'DisplayName', '$\Delta t = 100$ $\mu$s', ...
    'Color', color_100us, 'LineWidth', 3.5, 'MarkerSize', 6);
hold on;
plot(pscad_800us.t*1e3, pscad_800us.IL, 's-', 'DisplayName', '$\Delta t = 800$ $\mu$s', ...
    'Color', color_800us, 'LineWidth', 3.5, 'MarkerSize', 8);
grid on;
xlabel('Time [ms]', 'Interpreter', 'latex');
ylabel('Inductor Current $i(t)$ [A]', 'Interpreter', 'latex');
title('\textbf{Inductor Current}', 'Interpreter', 'latex', 'FontSize', 12, 'Color', 'k');
legend('Location', 'best', 'Interpreter', 'latex');
xlim([0 T_horizon_s*1e3]);
set(ax2, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'TickLabelInterpreter', 'latex');
% Explicit minor grid settings
ax2.XMinorGrid = 'on';
ax2.YMinorGrid = 'on';
ax2.MinorGridLineStyle = '-';
ax2.MinorGridAlpha = 0.15;
ax2.MinorGridColor = [0.5 0.5 0.5];

%% ===================== Save Plot =====================
figuresFolder = "../tex_Hw02/figures/";
if ~exist(figuresFolder, 'dir')
    mkdir(figuresFolder);
end

% Save figure
saveas(fig1, fullfile(figuresFolder, 'hw02_qB_pscad-comparison.png'));
fprintf('\nSaved: hw02_qB_pscad-comparison.png\n');

fprintf('\n=== Plotting Complete ===\n');
