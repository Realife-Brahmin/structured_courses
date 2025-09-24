T_horizon_s = 0.1;
h_ms = 0.1;
h_s = h_ms * 1e-3;


% model = fullfile(rawDataFolder, 'hw01_q03'); % without .slx extension
model_file = fullfile(rawDataFolder, 'hw01_q03.slx');
model_file = strrep(model_file, '\', '/');
[~, model, ~] = fileparts(model_file);

% Linearize the model to get state-space matrices
% [~, ~, ~, ~, xnames] = linmod(model);
% disp(xnames);
[A, B, C, D, xnames] = linmod(model);
stateNames = xnames.stateName;

disp(stateNames);
% Compute the eigenvalues of the A matrix
format long g
eigvals = eig(A);

% Display the eigenvalues
disp('Eigenvalues of the system:');
disp(eigvals);
% Set simulation start and stop time
set_param(model, 'StartTime', '0.0');
set_param(model, 'StopTime', num2str(T_horizon_s)); % T_horizon_s must be defined in workspace

% Set solver type and options
set_param(model, 'SolverType', 'Fixed-step'); % or 'Variable-step'
set_param(model, 'Solver', 'ode3'); % 'ode3' for Bogacki-Shampine, 'ode45', etc.
set_param(model, 'FixedStep', num2str(h_s)); % h_s must be defined in workspace



% Save the model if you want to persist these changes
% save_system(model);