% hw02_qA1.m

% Dynamically generate the folder path based on the script's location
currentScriptFolder = fileparts(mfilename('fullpath'));
cd(currentScriptFolder);
addpath(currentScriptFolder);

%% ===================== Setup Plot Theme =====================
setup_plot_theme();

% Superior color palette
color_trapz = [0.85, 0.33, 0.10];  % Burnt Orange (sophisticated)
color_BE = [0.00, 0.45, 0.74];     % Deep Blue (rich, professional)
color_pscad = [0.05, 0.28, 0.63];  % Dark PSCAD Blue (distinct from BE)

% --- Setup ---
T_horizon_s = 1e-3;
h_ms = 0.1;
h_s = h_ms * 1e-3;
h_mus = h_ms * 1e3;

% Circuit parameters
V_DC = 10;          % DC voltage [V]
V_0 = 0;            % Initial voltage [V] (can be changed)
C = 1e-6;           % Capacitance [F] (1 µF)
h = h_s;            % Time step [s]

% Discretization resistances
R_C_trapz = h / (2*C);      % Trapezoidal equivalent resistance
R_C_BE = h / C;              % Backward Euler equivalent resistance
R_epsilon = 1e-3 * R_C_BE;   % Forward Euler artificial resistance (very small)

% Time vector
t = 0:h:T_horizon_s;
N = length(t);

rawDataFolder = "./rawData/";
processedDataFolder = "../processedData/Hw02/";

%% ===================== Parse PSCAD Data =====================
fprintf('\n=== Parsing PSCAD Data ===\n');
pscad_folder = fullfile(processedDataFolder, 'hw02_qA_100mus');
pscad_data = parse_pscad_output(pscad_folder, 'hw02_qA_100mus');

% Display in table format similar to integration methods
fprintf('\n--- PSCAD Simulation Results ---\n');
fprintf('%-8s | %-12s | %-12s\n', 't [ms]', 'v_C [V]', 'i_C [A]');
fprintf('%s\n', repmat('-', 1, 40));
for idx = 1:length(pscad_data.t)
    fprintf('%-8.3f | %-12.6f | %-12.6f\n', ...
        pscad_data.t(idx)*1e3, ...  % Convert to ms
        pscad_data.vC(idx), ...
        pscad_data.IC(idx));
end
fprintf('\n');

%% ===================== Trapezoidal Method =====================
fprintf('\n=== Trapezoidal Method ===\n');
fprintf('R_C = %.6e Ω\n', R_C_trapz);

% Initialize arrays
v_trapz = zeros(1, N);
i_trapz = zeros(1, N);
e_h_trapz = zeros(1, N);
i_h_trapz = zeros(1, N);

% Initial condition at t = -h (index 1 represents t = 0)
v_trapz(1) = V_0;
i_trapz(1) = 0;

% Time step loop for trapezoidal
for n = 2:N
    % History terms from previous step
    % e_h = v(t-h) + R_C * i(t-h)
    % i_h = e_h / R_C  (Norton equivalent history current)
    e_h_trapz(n) = v_trapz(n-1) + R_C_trapz * i_trapz(n-1);
    i_h_trapz(n) = e_h_trapz(n) / R_C_trapz;
    
    % Current step: v(t) = V_DC (voltage source constraint)
    v_trapz(n) = V_DC;
    
    % i(t) = v(t)/R_C - i_h
    i_trapz(n) = v_trapz(n) / R_C_trapz - i_h_trapz(n);
end

%% ===================== Backward Euler Method =====================
fprintf('\n=== Backward Euler Method ===\n');
fprintf('R_C = %.6e Ω\n', R_C_BE);

% Initialize arrays
v_BE = zeros(1, N);
i_BE = zeros(1, N);
e_h_BE = zeros(1, N);
i_h_BE = zeros(1, N);

% Initial condition
v_BE(1) = V_0;
i_BE(1) = 0;

% Time step loop for BE
for n = 2:N
    % History terms from previous step
    % e_h = v(t-h) + R_C * i(t-h)  (but for BE, the R_C*i term is from previous step)
    % For BE: e_h = v(t-h), i_h = v(t-h)/R_C
    e_h_BE(n) = v_BE(n-1) + R_C_BE * i_BE(n-1);
    i_h_BE(n) = v_BE(n-1) / R_C_BE;
    
    % Current step: v(t) = V_DC (voltage source constraint)
    v_BE(n) = V_DC;
    
    % i(t) = v(t)/R_C - i_h
    i_BE(n) = v_BE(n) / R_C_BE - i_h_BE(n);
end

%% ===================== Forward Euler Method =====================
fprintf('\n=== Forward Euler Method ===\n');
fprintf('R_epsilon = %.6e Ω\n', R_epsilon);
fprintf('R_c = %.6e Ω\n', h/C);

% Initialize arrays (including -h step for reference)
v_FE = zeros(1, N+1);  % Extra element for t = -h
i_FE = zeros(1, N+1);
e_h_FE = zeros(1, N+1);
i_h_FE = zeros(1, N+1);

% Initial condition at t = -h (index 1)
v_FE(1) = V_0;  % V(-h) = V_0
i_FE(1) = 0;    % i(-h) = 0

% Forward Euler companion model using your corrected equations:
% v(t) = V_DC for t > 0 (KVL constraint)
% i(t) = (v(t) - e_h(t)) / R_epsilon               (Eq 6)
% where:
% e_h(t) = v(t-h) + R_c * i(t-h)
% i_h(t) = e_h(t) / R_c

R_c = h/C;  % Discretized capacitor resistance

% Time step loop for Forward Euler
for n = 2:N+1  % n=2 corresponds to t=0, n=3 to t=h, etc.
    % Step 1: v(t) is always V_DC for t > 0 (KVL constraint)
    v_FE(n) = V_DC;
    
    % Step 2: Calculate e_h and i_h using PREVIOUS row's v(t-h) and i(t-h)
    e_h_FE(n) = v_FE(n-1) + R_c * i_FE(n-1);
    i_h_FE(n) = e_h_FE(n) / R_epsilon;
    
    % Step 3: Calculate current i(t) using current row's v(t), e_h, i_h
    % From FE equation: i(t) = (v(t) - e_h(t)) / R_epsilon
    i_FE(n) = (v_FE(n) - e_h_FE(n)) / R_epsilon;
end

% Remove the -h step for plotting and table generation
v_FE = v_FE(2:end);  % Remove first element (t = -h)
i_FE = i_FE(2:end);
e_h_FE = e_h_FE(2:end);
i_h_FE = i_h_FE(2:end);

%% ===================== Generate Tables =====================
fprintf('\n=== Trapezoidal Method Table (First 5 steps) ===\n');
fprintf('%-10s %-15s %-15s %-15s %-15s\n', 't [ms]', 'v(t) [V]', 'i(t) [A]', 'e_h [V]', 'i_h [A]');
fprintf('%-10s %-15s %-15s %-15s %-15s\n', '----------', '---------------', '---------------', '---------------', '---------------');
for n = 1:min(6, N)
    fprintf('%-10.3f %-15.6e %-15.6e %-15.6e %-15.6e\n', ...
        t(n)*1e3, v_trapz(n), i_trapz(n), e_h_trapz(n), i_h_trapz(n));
end

fprintf('\n=== Backward Euler Method Table (First 5 steps) ===\n');
fprintf('%-10s %-15s %-15s %-15s %-15s\n', 't [ms]', 'v(t) [V]', 'i(t) [A]', 'e_h [V]', 'i_h [A]');
fprintf('%-10s %-15s %-15s %-15s %-15s\n', '----------', '---------------', '---------------', '---------------', '---------------');
for n = 1:min(6, N)
    fprintf('%-10.3f %-15.6e %-15.6e %-15.6e %-15.6e\n', ...
        t(n)*1e3, v_BE(n), i_BE(n), e_h_BE(n), i_h_BE(n));
end

fprintf('\n=== Forward Euler Method Table (First 5 steps) ===\n');
fprintf('%-10s %-15s %-15s %-15s %-15s\n', 't [ms]', 'v(t) [V]', 'i(t) [A]', 'e_h [V]', 'i_h [A]');
fprintf('%-10s %-15s %-15s %-15s %-15s\n', '----------', '---------------', '---------------', '---------------', '---------------');
for n = 1:min(6, N)
    fprintf('%-10.3f %-15.6e %-15.6e %-15.6e %-15.6e\n', ...
        t(n)*1e3, v_FE(n), i_FE(n), e_h_FE(n), i_h_FE(n));
end

%% ===================== Plot: Trapezoidal and BE Comparison =====================
fig1 = figure('Name', 'Trapezoidal vs Backward Euler', 'Position', [100 100 1200 500], 'Color', 'w');

% Add main title and subtitle with parameters
sgtitle({'\textbf{State Variables: Capacitor Circuit}', ...
    sprintf('$V_{\\mathrm{DC}} = %.0f$ V, $V_0 = %.0f$ V, $C = %.0f$ $\\mu$F, $h = %.0f$ $\\mu$s, $T = %.1f$ ms', ...
    V_DC, V_0, C*1e6, h*1e6, T_horizon_s*1e3)}, ...
    'Interpreter', 'latex', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

% Voltage comparison
subplot(1,2,1);
ax1 = gca;
plot(t*1e3, v_trapz, 'o-', 'DisplayName', 'Trapezoidal', 'Color', color_trapz, 'LineWidth', 3.5);
hold on;
plot(t*1e3, v_BE, 's-', 'DisplayName', 'Backward Euler', 'Color', color_BE, 'LineWidth', 3.5);
plot(pscad_data.t*1e3, pscad_data.vC, 'd--', 'DisplayName', 'PSCAD', 'Color', color_pscad, 'LineWidth', 3.2, 'MarkerSize', 7);
grid on;
xlabel('Time [ms]', 'Interpreter', 'latex');
ylabel('Voltage $v(t)$ [V]', 'Interpreter', 'latex');
title('\textbf{Voltage}', 'Interpreter', 'latex', 'FontSize', 12, 'Color', 'k');
legend('Location', 'best', 'Interpreter', 'latex');
xlim([0 T_horizon_s*1e3]);
set(ax1, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'TickLabelInterpreter', 'latex');
% Explicit minor grid settings
ax1.XMinorGrid = 'on';
ax1.YMinorGrid = 'on';
ax1.MinorGridLineStyle = '-';
ax1.MinorGridAlpha = 0.15;
ax1.MinorGridColor = [0.5 0.5 0.5];

% Current comparison
subplot(1,2,2);
ax2 = gca;
plot(t*1e3, i_trapz, 'o-', 'DisplayName', 'Trapezoidal', 'Color', color_trapz, 'LineWidth', 3.5);
hold on;
plot(t*1e3, i_BE, 's-', 'DisplayName', 'Backward Euler', 'Color', color_BE, 'LineWidth', 3.5);
plot(pscad_data.t*1e3, pscad_data.IC, 'd--', 'DisplayName', 'PSCAD', 'Color', color_pscad, 'LineWidth', 3.2, 'MarkerSize', 7);
grid on;
xlabel('Time [ms]', 'Interpreter', 'latex');
ylabel('Current $i(t)$ [A]', 'Interpreter', 'latex');
title('\textbf{Current}', 'Interpreter', 'latex', 'FontSize', 12, 'Color', 'k');
legend('Location', 'best', 'Interpreter', 'latex');
xlim([0 T_horizon_s*1e3]);
set(ax2, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'TickLabelInterpreter', 'latex');
% Explicit minor grid settings
ax2.XMinorGrid = 'on';
ax2.YMinorGrid = 'on';
ax2.MinorGridLineStyle = '-';
ax2.MinorGridAlpha = 0.15;
ax2.MinorGridColor = [0.5 0.5 0.5];

%% ===================== Plot: Forward Euler (Separate) =====================
fig2 = figure('Name', 'Forward Euler Method', 'Position', [150 150 1200 500], 'Color', 'w');

% Add main title for FE plot
sgtitle({'\textbf{Forward Euler Method: Capacitor Circuit}', ...
    sprintf('$V_{\\mathrm{DC}} = %.0f$ V, $V_0 = %.0f$ V, $C = %.0f$ $\\mu$F, $h = %.0f$ $\\mu$s, $T = %.1f$ ms', ...
    V_DC, V_0, C*1e6, h*1e6, T_horizon_s*1e3)}, ...
    'Interpreter', 'latex', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

% Voltage
subplot(1,2,1);
ax1 = gca;
plot(t*1e3, v_FE, 'o-', 'Color', [1 0.4 0.7], 'LineWidth', 3.5);
grid on;
xlabel('Time [ms]', 'Interpreter', 'latex');
ylabel('Voltage $v(t)$ [V]', 'Interpreter', 'latex');
title('\textbf{Voltage}', 'Interpreter', 'latex', 'FontSize', 12, 'Color', 'k');
xlim([0 T_horizon_s*1e3]);
set(ax1, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'TickLabelInterpreter', 'latex');
% Explicit minor grid settings
ax1.XMinorGrid = 'on';
ax1.YMinorGrid = 'on';
ax1.MinorGridLineStyle = '-';
ax1.MinorGridAlpha = 0.15;
ax1.MinorGridColor = [0.5 0.5 0.5];

% Current (symmetric log scale for diverging oscillations)
subplot(1,2,2);
ax2 = gca;

% Create symmetric log transformation: sign(x) * log10(abs(x)) but plot on linear scale
% with custom tick labels showing actual values
i_FE_symlog = sign(i_FE) .* log10(abs(i_FE) + 1e-10);

plot(t*1e3, i_FE_symlog, 'o-', 'Color', [1 0.4 0.7], 'LineWidth', 3.5);
grid on;
xlabel('Time [ms]', 'Interpreter', 'latex');
ylabel('Current $i(t)$ [A]', 'Interpreter', 'latex');
title('\textbf{Current (Note: Diverging oscillations)}', 'Interpreter', 'latex', 'FontSize', 12, 'Color', 'k');
xlim([0 T_horizon_s*1e3]);

% Set custom y-ticks based on actual data points
% The data oscillates with powers: 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32 (positive and negative)
% Each step multiplies by ~10^3, so we need higher powers
yticks([-32, -29, -26, -23, -20, -17, -14, -11, -8, -5, -2, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32]);
yticklabels({'-10^{32}', '-10^{29}', '-10^{26}', '-10^{23}', '-10^{20}', '-10^{17}', '-10^{14}', '-10^{11}', '-10^{8}', '-10^{5}', '-10^{2}', ...
             '10^{2}', '10^{5}', '10^{8}', '10^{11}', '10^{14}', '10^{17}', '10^{20}', '10^{23}', '10^{26}', '10^{29}', '10^{32}'});

set(ax2, 'XColor', 'k', 'YColor', 'k', 'Color', 'w', 'TickLabelInterpreter', 'tex');
% Explicit minor grid settings
ax2.XMinorGrid = 'on';
ax2.YMinorGrid = 'on';
ax2.MinorGridLineStyle = '-';
ax2.MinorGridAlpha = 0.15;
ax2.MinorGridColor = [0.5 0.5 0.5];

%% ===================== Save Plots =====================
figuresFolder = "../tex_Hw02/figures/";
if ~exist(figuresFolder, 'dir')
    mkdir(figuresFolder);
end

% Save Figure 1: State variable trajectory comparison (PNG only)
saveas(fig1, fullfile(figuresFolder, 'hw02_qA_state-variable-trajectory-comparison.png'));
fprintf('Saved: hw02_qA_state-variable-trajectory-comparison.png\n');

% Save Figure 2: Forward Euler
saveas(fig2, fullfile(figuresFolder, 'hw02_qA_forward-euler-method.png'));
fprintf('Saved: hw02_qA_forward-euler-method.png\n');

fprintf('\n=== Simulation Complete ===\n');

