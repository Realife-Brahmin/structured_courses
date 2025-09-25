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

model_file = fullfile(rawDataFolder, 'hw01_Fig1b.slx');
model_file = strrep(model_file, '\', '/');
[~, model, ~] = fileparts(model_file);

T_horizon_s = 0.2;
% Only use nonstiff and stiff variable-step solvers
stiff_solvers = ["ode23s"];
nonstiff_solvers = ["ode23"];
varstep_solvers = [stiff_solvers, nonstiff_solvers];
solver_names = [nonstiff_solvers, stiff_solvers];

Vc_results = cell(1, numel(varstep_solvers));
iL1_results = cell(1, numel(varstep_solvers));
iL2_results = cell(1, numel(varstep_solvers));
t_results = cell(1, numel(varstep_solvers));

% Settings to match q06
figure('Color', 'w');
colors = lines(numel(varstep_solvers));

% Simulate variable-step solvers (nonstiff and stiff)
for k = 1:numel(varstep_solvers)
    solver = varstep_solvers(k);
    set_param(model, 'StartTime', '0.0');
    set_param(model, 'StopTime', num2str(T_horizon_s));
    set_param(model, 'SolverType', 'Variable-step');
    set_param(model, 'Solver', solver);
    simOut = sim(model, 'ReturnWorkspaceOutputs', 'on');
    Vc_results{k} = simOut.V_c_t.Data;
    iL1_results{k} = simOut.i_L1_t.Data * 1000;
    iL2_results{k} = simOut.i_L2_t.Data * 1000;
    t_results{k}  = simOut.V_c_t.Time;
end

% Simulate fixed-step solvers for table only
fixedstep_solvers = ["ode14x"];
h_s_list = [1e-4];
for k = 1:numel(fixedstep_solvers)
    solver = fixedstep_solvers(k);
    h_s = h_s_list(k);
    set_param(model, 'StartTime', '0.0');
    set_param(model, 'StopTime', num2str(T_horizon_s));
    set_param(model, 'SolverType', 'Fixed-step');
    set_param(model, 'Solver', solver);
    set_param(model, 'FixedStep', num2str(h_s));
    simOut = sim(model, 'ReturnWorkspaceOutputs', 'on');
    Vc_results{numel(varstep_solvers) + k} = simOut.V_c_t.Data;
    iL1_results{numel(varstep_solvers) + k} = simOut.i_L1_t.Data * 1000;
    iL2_results{numel(varstep_solvers) + k} = simOut.i_L2_t.Data * 1000;
    t_results{numel(varstep_solvers) + k}  = simOut.V_c_t.Time;
end

% Plotting (match q06: 2 subplots, only varstep solvers)
subplot(2,1,1);
hold on;
for k = 1:numel(varstep_solvers)
    plot(t_results{k}, iL1_results{k}, 'Color', colors(k,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Solver = %s', varstep_solvers(k)));
end
ylabel('i_{L1}(t) [A]', 'Color', 'k');
title('Inductor Current i_{L1}', 'Interpreter', 'latex', 'Color', 'k');
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
for k = 1:numel(varstep_solvers)
    plot(t_results{k}, Vc_results{k}, 'Color', colors(k,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Solver = %s', varstep_solvers(k)));
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

% Table summarizing solver performance (including fixed-step)
all_solver_names = [varstep_solvers, fixedstep_solvers];
summaryTableB = table(all_solver_names(:), ...
    cellfun(@(t) t(end), t_results(:)), ...
    cellfun(@(t) numel(t), t_results(:)), ...
    'VariableNames', {'Solver', 'FinalTime_s', 'NumSteps'});
disp(summaryTableB);
writetable(summaryTableB, 'hw01_q11_solver_summary_B.csv');