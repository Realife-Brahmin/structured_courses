%% ===================== Global light theme =====================
set(groot,'defaultFigureColor','w');
set(groot,'defaultAxesColor','w');
set(groot,'defaultAxesXColor','k');
set(groot,'defaultAxesYColor','k');
set(groot,'defaultAxesGridColor',[.2 .2 .2]);
set(groot,'defaultAxesMinorGridColor',[.6 .6 .6]);
set(groot,'defaultAxesFontName','Helvetica');
set(groot,'defaultAxesFontSize',12);
set(groot,'defaultAxesTitleFontWeight','bold');
set(groot,'defaultAxesTitleFontSizeMultiplier',1.1);
set(groot,'defaultAxesXMinorGrid','on');
set(groot,'defaultAxesYMinorGrid','on');
set(groot,'defaultLineLineWidth',1.4);
set(groot,'defaultLineMarkerSize',9);
set(groot, 'defaultLegendColor', 'w');      % White background for all legends
set(groot, 'defaultLegendTextColor', 'k');  % Black text for all legends

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

% %% ===================== Forward Euler Method =====================
% fprintf('\n=== Forward Euler Method ===\n');
% fprintf('R_epsilon = %.6e Ω\n', R_epsilon);
% fprintf('R_epsilon/R_C_BE ratio = %.6e\n', R_epsilon/R_C_BE);
% 
% % Initialize arrays
% v_FE = zeros(1, N);
% i_FE = zeros(1, N);
% e_h_FE = zeros(1, N);
% i_h_FE = zeros(1, N);
% 
% % Initial condition
% v_FE(1) = V_0;
% i_FE(1) = 0;
% 
% % Define R_c for FE (using same as BE for consistency)
% R_c_FE = R_C_BE;
% 
% % Time step loop for FE
% for n = 2:N
%     if n == 2
%         % First step
%         e_h_FE(n) = V_0;
%         i_h_FE(n) = e_h_FE(n) / R_epsilon;
%         v_FE(n) = e_h_FE(n);
%         i_FE(n) = (V_DC - e_h_FE(n)) / R_epsilon;
%     else
%         % History terms
%         e_h_FE(n) = v_FE(n-1) + R_c_FE * i_FE(n-1);
%         i_h_FE(n) = e_h_FE(n) / R_epsilon;
%         % Current step
%         v_FE(n) = e_h_FE(n);
%         i_FE(n) = (V_DC - e_h_FE(n)) / R_epsilon;
%     end
% end

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

% fprintf('\n=== Forward Euler Method Table (First 5 steps) ===\n');
% fprintf('%-10s %-15s %-15s %-15s %-15s\n', 't [ms]', 'v(t) [V]', 'i(t) [A]', 'e_h [V]', 'i_h [A]');
% fprintf('%-10s %-15s %-15s %-15s %-15s\n', '----------', '---------------', '---------------', '---------------', '---------------');
% for n = 1:min(6, N)
%     fprintf('%-10.3f %-15.6e %-15.6e %-15.6e %-15.6e\n', ...
%         t(n)*1e3, v_FE(n), i_FE(n), e_h_FE(n), i_h_FE(n));
% end

%% ===================== Plot: Trapezoidal and BE Comparison =====================
fig1 = figure('Name', 'Trapezoidal vs Backward Euler', 'Position', [100 100 1200 500], 'Color', 'w');

% Voltage comparison
subplot(1,2,1);
ax1 = gca;
plot(t*1e3, v_trapz, 'o-', 'DisplayName', 'Trapezoidal', 'Color', [1 0.5 0]);
hold on;
plot(t*1e3, v_BE, 's-', 'DisplayName', 'Backward Euler', 'Color', [0.2 0.5 0.8]);
grid on;
xlabel('Time [ms]', 'Color', 'k');
ylabel('Voltage v(t) [V]', 'Color', 'k');
title('Voltage: Trapezoidal vs BE', 'Color', 'k');
legend('Location', 'best');
xlim([0 T_horizon_s*1e3]);
set(ax1, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');

% Current comparison
subplot(1,2,2);
ax2 = gca;
plot(t*1e3, i_trapz, 'o-', 'DisplayName', 'Trapezoidal', 'Color', [1 0.5 0]);
hold on;
plot(t*1e3, i_BE, 's-', 'DisplayName', 'Backward Euler', 'Color', [0.2 0.5 0.8]);
grid on;
xlabel('Time [ms]', 'Color', 'k');
ylabel('Current i(t) [A]', 'Color', 'k');
title('Current: Trapezoidal vs BE', 'Color', 'k');
legend('Location', 'best');
xlim([0 T_horizon_s*1e3]);
set(ax2, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');

% %% ===================== Plot: Forward Euler (Separate) =====================
% fig2 = figure('Name', 'Forward Euler Method', 'Position', [150 150 1200 500], 'Color', 'w');
% 
% % Voltage
% subplot(1,2,1);
% ax1 = gca;
% plot(t*1e3, v_FE, 'o-', 'Color', [1 0.4 0.7]);
% grid on;
% xlabel('Time [ms]', 'Color', 'k');
% ylabel('Voltage v(t) [V]', 'Color', 'k');
% title('FE: Voltage', 'Color', 'k');
% xlim([0 T_horizon_s*1e3]);
% set(ax1, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');
% 
% % Current (scaled y-axis for large values)
% subplot(1,2,2);
% ax2 = gca;
% plot(t*1e3, i_FE, 'o-', 'Color', [1 0.4 0.7]);
% grid on;
% xlabel('Time [ms]', 'Color', 'k');
% ylabel('Current i(t) [A]', 'Color', 'k');
% title('FE: Current (Note: Large oscillations)', 'Color', 'k');
% xlim([0 T_horizon_s*1e3]);
% set(ax2, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');

%% ===================== Save Plots =====================
figuresFolder = "../tex_Hw02/figures/";
if ~exist(figuresFolder, 'dir')
    mkdir(figuresFolder);
end

% Save Figure 1: Trapezoidal vs BE (PNG only)
saveas(fig1, fullfile(figuresFolder, 'hw02_qA1_trapz_vs_BE.png'));
fprintf('Saved: hw02_qA1_trapz_vs_BE.png\n');

% % Save Figure 2: Forward Euler
% saveas(fig2, fullfile(figuresFolder, 'hw02_qA1_FE.png'));
% fprintf('Saved: hw02_qA1_FE.png\n');

fprintf('\n=== Simulation Complete ===\n');

