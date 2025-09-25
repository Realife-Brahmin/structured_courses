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

T_horizon_s = 0.2;
% ode_solvers = ["ode5", "ode8"];
% h_ms_list = [0.5, 1.6]; % converging
ode_solvers = ["ode5"];
h_ms_list = [0.1]; % good
% h_ms_list = [1.0]; % diverging
h_s_list = h_ms_list * 1e-3;

Vc_results = cell(1, numel(h_s_list));
iL_results = cell(1, numel(h_s_list));
t_results = cell(1, numel(h_s_list));

for k = 1:numel(h_s_list)
    h_s = h_s_list(k);
    set_param(model, 'StartTime', '0.0');
    set_param(model, 'StopTime', num2str(T_horizon_s));
    set_param(model, 'SolverType', 'Fixed-step');
    set_param(model, 'Solver', ode_solvers(k)); % Dormand-Prince RK5
    set_param(model, 'FixedStep', num2str(h_s));
    
    simOut = sim(model, 'ReturnWorkspaceOutputs', 'on');
    % Adjust these variable names if your To Workspace blocks use different names
    Vc_results{k} = simOut.V_c_t.Data;
    iL_results{k} = simOut.i_L_t.Data * 1000; % Convert kA to A
    t_results{k}  = simOut.V_c_t.Time;
end

% Plotting
figure('Color', 'w');
colors = lines(numel(h_s_list));

subplot(2,1,1);
hold on;
for k = 1:numel(h_s_list)
    plot(t_results{k}, iL_results{k}, 'Color', colors(k,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('h = %.1f ms, solver = %s', h_ms_list(k), ode_solvers(k)));
end
ylabel('i_L(t) [A]', 'Color', 'k');
title('Inductor Current', 'Interpreter', 'latex', 'Color', 'k');
set(gca, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');
grid on; grid minor;
ax1 = gca;
ax1.MinorGridLineStyle = '-';
ax1.MinorGridAlpha = 0.18;
ax1.MinorGridColor = [0.5 0.5 0.5];
legend('show', 'TextColor', 'k', 'Location', 'best');
hold off;

subplot(2,1,2);
hold on;
for k = 1:numel(h_s_list)
    plot(t_results{k}, Vc_results{k}, 'Color', colors(k,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('h = %.1f ms, solver = %s', h_ms_list(k), ode_solvers(k)));
end
ylabel('V_C(t) [kV]', 'Color', 'k');
xlabel('Time (s)', 'Color', 'k');
title('Capacitor Voltage',  'Interpreter', 'latex', 'Color', 'k');
set(gca, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');
grid on; grid minor;
ax2 = gca;
ax2.MinorGridLineStyle = '-';
ax2.MinorGridAlpha = 0.18;
ax2.MinorGridColor = [0.5 0.5 0.5];
legend('show', 'TextColor', 'k', 'Location', 'best');
hold off;