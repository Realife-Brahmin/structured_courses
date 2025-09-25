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
h_max = 1e-2;
tol = 1e-3;
% Only use nonstiff and stiff variable-step solvers
stiff_solvers = ["ode15s", "ode23s", "ode23t", "ode23tb"];
% stiff_solvers = ["ode23s"];
nonstiff_solvers = ["ode23", "ode45", "ode113"];
varstep_solvers = [stiff_solvers, nonstiff_solvers];
solver_names = [nonstiff_solvers, stiff_solvers];

Vc_results = cell(1, numel(varstep_solvers));
iL1_results = cell(1, numel(varstep_solvers));
iL2_results = cell(1, numel(varstep_solvers));
t_results = cell(1, numel(varstep_solvers));

% Simulate variable-step solvers (nonstiff and stiff)
for k = 1:numel(varstep_solvers)
    solver = varstep_solvers(k);
    set_param(model, 'StartTime', '0.0');
    set_param(model, 'StopTime', num2str(T_horizon_s));
    set_param(model, 'SolverType', 'Variable-step');
    set_param(model, 'Solver', solver);
    set_param(model, 'MaxStep', num2str(h_max));
    set_param(model, 'RelTol', num2str(tol));
    set_param(model, 'AbsTol', num2str(tol));
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

% Plotting: 6 subplots (3 for nonstiff, 3 for stiff), support multiple solvers per category
figure('Color', 'w');
nonstiff_idx = find(ismember(varstep_solvers, nonstiff_solvers));
stiff_idx = find(ismember(varstep_solvers, stiff_solvers));
colors = lines(max(numel(nonstiff_idx), numel(stiff_idx)));

% Nonstiff (left column)
subplot(3,2,1); % Vc nonstiff
hold on;
for idx = 1:length(nonstiff_idx)
    k = nonstiff_idx(idx);
    plot(t_results{k}, Vc_results{k}, 'Color', colors(idx,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Solver = %s', varstep_solvers(k)));
end
ylabel('V_C(t) [kV]', 'Color', 'k');
title('Capacitor Voltage (nonstiff)', 'Interpreter', 'latex', 'Color', 'k');
set(gca, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');
grid on; grid minor; legend('show', 'TextColor', 'k', 'Location', 'best');
hold off;

subplot(3,2,3); % iL1 nonstiff
hold on;
for idx = 1:length(nonstiff_idx)
    k = nonstiff_idx(idx);
    plot(t_results{k}, iL1_results{k}, 'Color', colors(idx,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Solver = %s', varstep_solvers(k)));
end
ylabel('i_{L1}(t) [A]', 'Color', 'k');
title('Inductor Current $i_{L1}$ (nonstiff)', 'Interpreter', 'latex', 'Color', 'k');
set(gca, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');
grid on; grid minor; legend('show', 'TextColor', 'k', 'Location', 'best');
hold off;

subplot(3,2,5); % iL2 nonstiff
hold on;
for idx = 1:length(nonstiff_idx)
    k = nonstiff_idx(idx);
    plot(t_results{k}, iL2_results{k}, 'Color', colors(idx,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Solver = %s', varstep_solvers(k)));
end
ylabel('i_{L2}(t) [A]', 'Color', 'k');
xlabel('Time (s)', 'Color', 'k');
title('Inductor Current $i_{L2}$ (nonstiff)', 'Interpreter', 'latex', 'Color', 'k');
set(gca, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');
grid on; grid minor; legend('show', 'TextColor', 'k', 'Location', 'best');
hold off;

% Stiff (right column)
subplot(3,2,2); % Vc stiff
hold on;
for idx = 1:length(stiff_idx)
    k = stiff_idx(idx);
    plot(t_results{k}, Vc_results{k}, 'Color', colors(idx,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Solver = %s', varstep_solvers(k)));
end
ylabel('V_C(t) [kV]', 'Color', 'k');
title('Capacitor Voltage (stiff)', 'Interpreter', 'latex', 'Color', 'k');
set(gca, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');
grid on; grid minor; legend('show', 'TextColor', 'k', 'Location', 'best');
hold off;

subplot(3,2,4); % iL1 stiff
hold on;
for idx = 1:length(stiff_idx)
    k = stiff_idx(idx);
    plot(t_results{k}, iL1_results{k}, 'Color', colors(idx,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Solver = %s', varstep_solvers(k)));
end
ylabel('i_{L1}(t) [A]', 'Color', 'k');
title('Inductor Current $i_{L1}$ (stiff)', 'Interpreter', 'latex', 'Color', 'k');
set(gca, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');
grid on; grid minor; legend('show', 'TextColor', 'k', 'Location', 'best');
hold off;

subplot(3,2,6); % iL2 stiff
hold on;
for idx = 1:length(stiff_idx)
    k = stiff_idx(idx);
    plot(t_results{k}, iL2_results{k}, 'Color', colors(idx,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Solver = %s', varstep_solvers(k)));
end
ylabel('i_{L2}(t) [A]', 'Color', 'k');
xlabel('Time (s)', 'Color', 'k');
title('Inductor Current $i_{L2}$ (stiff)', 'Interpreter', 'latex', 'Color', 'k');
set(gca, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');
grid on; grid minor; legend('show', 'TextColor', 'k', 'Location', 'best');
hold off;

% --- Table with compute time and step info (no RMSE, as no analytical solution) ---
solver_names = [varstep_solvers, fixedstep_solvers]';
solver_types = repmat("Nonstiff", size(solver_names));
solver_types(ismember(solver_names, stiff_solvers)) = "Stiff";
solver_types(ismember(solver_names, fixedstep_solvers)) = "Fixed-step";

max_step_used = zeros(length(solver_names),1);
num_steps = zeros(length(solver_names),1);
sim_time = zeros(length(solver_names),1);

for k = 1:length(solver_names)
    t_solver = t_results{k};
    max_step_used(k) = max(diff(t_solver));
    num_steps(k) = length(t_solver);
    % If timing wasn't captured, rerun with tic/toc
    if k <= numel(varstep_solvers)
        solver = varstep_solvers(k);
        set_param(model, 'StartTime', '0.0');
        set_param(model, 'StopTime', num2str(T_horizon_s));
        set_param(model, 'SolverType', 'Variable-step');
        set_param(model, 'Solver', solver);
        tic;
        sim(model, 'ReturnWorkspaceOutputs', 'on');
        sim_time(k) = toc;
    else
        solver = fixedstep_solvers(k-numel(varstep_solvers));
        h_s = h_s_list(k-numel(varstep_solvers));
        set_param(model, 'StartTime', '0.0');
        set_param(model, 'StopTime', num2str(T_horizon_s));
        set_param(model, 'SolverType', 'Fixed-step');
        set_param(model, 'Solver', solver);
        set_param(model, 'FixedStep', num2str(h_s));
        tic;
        sim(model, 'ReturnWorkspaceOutputs', 'on');
        sim_time(k) = toc;
    end
end

T = table(solver_names, solver_types, max_step_used, num_steps, sim_time, ...
    'VariableNames', {'Solver', 'Type', 'MaxStepUsed', 'NumSteps', 'SimTime_sec'});
disp(T);
writetable(T, 'hw01_q11_solver_summary_B.csv');