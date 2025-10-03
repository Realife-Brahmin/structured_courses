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

model_file = fullfile(rawDataFolder, 'hw01_Fig1a.slx');
model_file = strrep(model_file, '\', '/');
[~, model, ~] = fileparts(model_file);
load_system(model);

T_horizon_s = 0.2;
h_max = 1e-2;
tol = 1e-3;

nonstiff_solvers = {'ode45', 'ode23', 'ode113'};
% nonstiff_solvers = {'ode45', 'ode23'};
stiff_solvers = {'ode15s', 'ode23s', 'ode23t', 'ode23tb'};
% stiff_solvers = {'ode23s', 'ode23t', 'ode23tb'};
var_step_solvers = [nonstiff_solvers, stiff_solvers];
results = struct();

for k = 1:length(var_step_solvers)
    solver = var_step_solvers{k};
    set_param(model, 'StartTime', '0.0');
    set_param(model, 'StopTime', num2str(T_horizon_s));
    set_param(model, 'SolverType', 'Variable-step');
    set_param(model, 'Solver', solver);
    set_param(model, 'MaxStep', num2str(h_max));
    set_param(model, 'RelTol', num2str(tol));
    set_param(model, 'AbsTol', num2str(tol));

    tic;
    simOut = sim(model, 'ReturnWorkspaceOutputs', 'on');
    elapsed = toc;

    t = simOut.V_c_t.Time;
    Vc = simOut.V_c_t.Data;
    iL = simOut.i_L_t.Data * 1000; % kA to A

    results(k).solver = solver;
    results(k).t = t;
    results(k).Vc = Vc;
    results(k).iL = iL;
    results(k).maxStepUsed = max(diff(t));
    results(k).numSteps = length(t);
    results(k).simTime = elapsed;
end

fixed_step_solvers = {'ode5', 'ode8'};
solvers_all = [var_step_solvers, fixed_step_solvers]
% Added RK5 and RK8 solvers with fixed-step settings
fixed_step_size = 0.5e-3; % Fixed step size for RK5 and RK8

for k = 1:length(fixed_step_solvers)
    solver = fixed_step_solvers{k};
    set_param(model, 'StartTime', '0.0');
    set_param(model, 'StopTime', num2str(T_horizon_s));
    set_param(model, 'SolverType', 'Fixed-step');
    set_param(model, 'Solver', solver);
    set_param(model, 'FixedStep', num2str(fixed_step_size));

    tic;
    simOut = sim(model, 'ReturnWorkspaceOutputs', 'on');
    elapsed = toc;

    t = simOut.V_c_t.Time;
    Vc = simOut.V_c_t.Data;
    iL = simOut.i_L_t.Data * 1000; % kA to A

    results(end+1).solver = solver;
    results(end).t = t;
    results(end).Vc = Vc;
    results(end).iL = iL;
    results(end).maxStepUsed = max(diff(t));
    results(end).numSteps = length(t);
    results(end).simTime = elapsed;
end

% Table with compute time
solver_names = {results.solver}';
max_step_used = [results.maxStepUsed]';
num_steps = [results.numSteps]';
sim_time = [results.simTime]';

% Updated solver categorization to include fixed-step solvers
solver_types = repmat("Nonstiff", size(solver_names));
solver_types(ismember(solver_names, stiff_solvers)) = "Stiff";
solver_types(ismember(solver_names, fixed_step_solvers)) = "Fixed-step";

T = table(solver_names, solver_types, max_step_used, num_steps, sim_time, ...
    'VariableNames', {'Solver', 'Type', 'MaxStepUsed', 'NumSteps', 'SimTime_sec'});

% Load or compute analytical solution
t_analytical = t; % from main.m
x_analytical = x_t_kV; % from main.m

% Compute error for all solvers
error_col = zeros(length(results),1);

for k = 1:length(results)
    t_solver = results(k).t;
    Vc_solver = results(k).Vc;
    % Evaluate analytical solution at solver's time points
    x_analytical_at_solver = x_ss_kV * (1 - (1/sqrt(1-zeta^2)) * exp(-zeta*w_n*t_solver) .* sin(w_d*t_solver + phi));
    % Compute RMSE or norm
    error_col(k) = sqrt(mean((Vc_solver - x_analytical_at_solver).^2));
end

% Add error to table
T.Error_kV = error_col;
disp(T);
writetable(T, 'variable_step_solver_summary.csv');

% Find indices for each group
nonstiff_idx = find(ismember({results.solver}, nonstiff_solvers));
stiff_idx = find(ismember({results.solver}, stiff_solvers));

figure('Color', 'w');
colors = lines(length(solvers));

% Nonstiff solvers
subplot(2,2,1);
hold on;
for idx = 1:length(nonstiff_idx)
    k = nonstiff_idx(idx);
    plot(results(k).t, results(k).iL, 'LineWidth', 1.5, 'DisplayName', results(k).solver, 'Color', colors(k,:));
end
ylabel('i_L(t) [A]', 'Color', 'k');
title('Inductor Current (Nonstiff)', 'Interpreter', 'latex', 'FontSize', 16, 'Color', 'k');
ax = gca;
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
legend('show', 'TextColor', 'k', 'Location', 'best');
grid on; grid minor; hold off;

subplot(2,2,3);
hold on;
for idx = 1:length(nonstiff_idx)
    k = nonstiff_idx(idx);
    plot(results(k).t, results(k).Vc, 'LineWidth', 1.5, 'DisplayName', results(k).solver, 'Color', colors(k,:));
end
ylabel('V_C(t) [kV]', 'Color', 'k');
xlabel('Time (s)', 'Color', 'k');
title('Capacitor Voltage (Nonstiff)', 'Interpreter', 'latex', 'FontSize', 16, 'Color', 'k');
ax = gca;
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
legend('show', 'TextColor', 'k', 'Location', 'best');
grid on; grid minor; hold off;

% Stiff solvers
subplot(2,2,2);
hold on;
for idx = 1:length(stiff_idx)
    k = stiff_idx(idx);
    plot(results(k).t, results(k).iL, 'LineWidth', 1.5, 'DisplayName', results(k).solver, 'Color', colors(k,:));
end
ylabel('i_L(t) [A]', 'Color', 'k');
title('Inductor Current (Stiff)', 'Interpreter', 'latex', 'FontSize', 16, 'Color', 'k');
ax = gca;
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
legend('show', 'TextColor', 'k', 'Location', 'best');
grid on; grid minor; hold off;

subplot(2,2,4);
hold on;
for idx = 1:length(stiff_idx)
    k = stiff_idx(idx);
    plot(results(k).t, results(k).Vc, 'LineWidth', 1.5, 'DisplayName', results(k).solver, 'Color', colors(k,:));
end
ylabel('V_C(t) [kV]', 'Color', 'k');
xlabel('Time (s)', 'Color', 'k');
title('Capacitor Voltage (Stiff)', 'Interpreter', 'latex', 'FontSize', 16, 'Color', 'k');
ax = gca;
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
legend('show', 'TextColor', 'k', 'Location', 'best');
grid on; grid minor; hold off;