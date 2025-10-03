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
T_horizon_s = 0.1;
h_ms = 0.1;
h_s = h_ms * 1e-3;

rawDataFolder = "./rawData/";
model_file = fullfile(rawDataFolder, 'hw01_Fig1a.slx');
model_file = strrep(model_file, '\', '/');
[~, model] = fileparts(model_file);
load_system(model);

% Linearize the model to get state-space matrices and eigenvalues
[A, B, C, D, xnames] = linmod(model);
stateNames = xnames.stateName;
disp('State names:');
disp(stateNames);

format long g
eigvals = eig(A);
disp('Eigenvalues of the system:');
disp(eigvals);

% Set simulation parameters
set_param(model, 'StartTime', '0.0');
set_param(model, 'StopTime', num2str(T_horizon_s));
set_param(model, 'SolverType', 'Fixed-step');
set_param(model, 'Solver', 'ode3'); % Bogacki-Shampine
set_param(model, 'FixedStep', num2str(h_s));

% Run simulation
simOut = sim(model, 'ReturnWorkspaceOutputs', 'on');
iL = simOut.i_L_t.Data * 1000; % Convert kA to A
Vc = simOut.V_c_t.Data;
t = simOut.i_L_t.Time;

% --- Plotting ---
figure('Color', 'w');

subplot(2,1,1);
ax1 = gca;
plot(t, iL, 'b', 'LineWidth', 1.5);
ylabel('i_L(t) [A]', 'Color', 'k');
title('Inductor Current', 'Color', 'k');
set(ax1, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');
grid on;
ax1.MinorGridLineStyle = '-';
ax1.MinorGridAlpha = 0.18;
ax1.MinorGridColor = [0.5 0.5 0.5];
grid minor;

subplot(2,1,2);
ax2 = gca;
plot(t, Vc, 'r', 'LineWidth', 1.5);
ylabel('V_C(t) [kV]', 'Color', 'k');
xlabel('Time (s)', 'Color', 'k');
title('Capacitor Voltage', 'Color', 'k');
set(ax2, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');
grid on;
ax2.MinorGridLineStyle = '-';
ax2.MinorGridAlpha = 0.18;
ax2.MinorGridColor = [0.5 0.5 0.5];
grid minor;