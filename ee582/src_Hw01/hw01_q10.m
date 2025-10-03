%% ===================== Global light theme (same as yours) =====================
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
set(groot, 'defaultLegendColor', 'w');
set(groot, 'defaultLegendTextColor', 'k');

format long g

%% ===================== Parameters (your values) =====================
r_L = 1;      L = 19e-3;     C_cap = 8.2e-6;
r_L2 = 323e-3; r_L1 = r_L - r_L2;
L_2  = 1e-3;   L_1  = L - L_2;
% r_C  = 1e6;
% r_C = 1e9;
r_C = 1e12;
% r_leak = 30e3;
% r_leak = 30e6;
r_leak = 30e9;

%% ===================== Hand state-space (Fig. 1b) =====================
% States x = [i_L1; i_L2; v_C], input u = v_dc
rSigma = r_leak;

A_hand = [ -(r_L1+rSigma)/L_1,   rSigma/L_1,             0;
            rSigma/L_2,         -(rSigma+r_L2)/L_2,  -1/L_2;
            0,                   1/C_cap,          -1/(r_C*C_cap) ];

B_hand = [1/L_1; 0; 0];      % input u = v_dc

% Eigenvalues from the hand matrix
lambda_hand = eig(A_hand);

disp('A_hand ='); disp(A_hand);
disp('B_hand ='); disp(B_hand);
disp('Eigenvalues from A_hand:');
disp(lambda_hand);

%% ===================== Linearize Simulink model and compare ==============
% Point to your Fig. 1(b) model (change the file name if needed)
rawDataFolder = "./rawData/";
model_file_b  = fullfile(rawDataFolder, 'hw01_Fig1b.slx');  % <--- rename if different
model_file_b  = strrep(model_file_b, '\','/');
[~, model_b]  = fileparts(model_file_b);

if exist(model_file_b,'file')
    % Load once so linmod can find it
    load_system(model_b);
    [A_lin, B_lin, C_lin, D_lin, xnames] = linmod(model_b);
    disp('r_leak');  disp(r_leak);
    disp('r_C');  disp(r_C);

    disp('State names (Simulink linmod):');  disp(xnames.stateName);
    disp('Eigenvalues from linmod(A_lin):'); disp(eig(A_lin));

    % Quick side-by-side comparison
    fprintf('\nComparison (sorted by real part):\n');
    LH = sortrows([real(lambda_hand), imag(lambda_hand)],1);
    LL = sortrows([real(eig(A_lin)),   imag(eig(A_lin))],1);
    T  = table(LH(:,1),LH(:,2),LL(:,1),LL(:,2), ...
        'VariableNames',{'Re(hand)','Im(hand)','Re(linmod)','Im(linmod)'});
    disp(T);
else
    warning('Simulink model not found at: %s\n(Skipping linmod comparison.)', model_file_b);
end

%% ===================== Suggest a fixed-step & run with a stiff solver ====
% Use a stiff, fixed-step solver (implicit): ode14x is a good choice here.
% Pick step from eigenvalues: ~T/20 of the fastest oscillatory mode OR 0.1/|Re(λ)| for real poles.
lam = lambda_hand;
omega = max(abs(imag(lam)));
if omega > 0
    h_from_imag = (2*pi/omega)/20;   % 20 samples per cycle
else
    h_from_imag = inf;
end
h_from_real = 0.1 / max(abs(real(lam)));  % 0.1x the fastest real time constant
h_s = min(h_from_imag, h_from_real);
h_s = min(h_s, 1e-3);                    % cap at 1 ms for safety; adjust if you like
fprintf('\nSuggested fixed step h = %.6g s (%.3f ms)\n', h_s, 1e3*h_s);
% h_s = 1e-5;
h_s = 1e-4;
if exist('model_b','var') && bdIsLoaded(model_b)
    set_param(model_b, 'StartTime','0.0');
    set_param(model_b, 'StopTime','0.2');        % adjust as needed
    set_param(model_b, 'SolverType','Fixed-step');
    solver = 'ode14x';
    set_param(model_b, 'Solver', solver);       % stiff, implicit fixed-step
    set_param(model_b, 'FixedStep', num2str(h_s));

    % Example run (requires logged signals in model):
    simOut = sim(model_b, 'ReturnWorkspaceOutputs','on');
    % Example signal names (change to yours):
    t   = simOut.V_c_t.Time;
    v_c = simOut.V_c_t.Data;
    i1  = simOut.i_L1_t.Data;
    i2  = simOut.i_L2_t.Data;
    % figure; plot(t, v_c); grid on; title('v_C(t)'); xlabel('Time (s)'); ylabel('V');
end

% --- Extract signals ---
t   = simOut.V_c_t.Time;
v_c = simOut.V_c_t.Data;         % [V]
i1  = simOut.i_L1_t.Data;        % [A]
i2  = simOut.i_L2_t.Data;        % [A]

% --- Plotting ---
figure('Color','w');

subplot(3,1,1);
ax1 = gca;
plot(t, i1*1e3, 'b', 'LineWidth', 1.5);
ylabel('i_{L1}(t) [A]', 'Color', 'k');
title('Inductor Current L1', 'Color', 'k');
subtitle(sprintf('h_s = %.1e s, Solver = %s', h_s, solver));

set(ax1,'XColor','k','YColor','k','Color','w');
grid on;
ax1.MinorGridLineStyle = '-';
ax1.MinorGridAlpha = 0.18;
ax1.MinorGridColor = [0.5 0.5 0.5];
grid minor;

subplot(3,1,2);
ax2 = gca;
plot(t, i2*1e3, 'm', 'LineWidth', 1.5);
ylabel('i_{L2}(t) [A]', 'Color', 'k');
title('Inductor Current L2', 'Color', 'k');
set(ax2,'XColor','k','YColor','k','Color','w');
grid on;
ax2.MinorGridLineStyle = '-';
ax2.MinorGridAlpha = 0.18;
ax2.MinorGridColor = [0.5 0.5 0.5];
grid minor;

subplot(3,1,3);
ax3 = gca;
plot(t, v_c, 'r', 'LineWidth', 1.5); % converted to kV
ylabel('v_C(t) [kV]', 'Color', 'k');
xlabel('Time (s)', 'Color', 'k');
title('Capacitor Voltage', 'Color', 'k');
set(ax3,'XColor','k','YColor','k','Color','w');
grid on;
ax3.MinorGridLineStyle = '-';
ax3.MinorGridAlpha = 0.18;
ax3.MinorGridColor = [0.5 0.5 0.5];
grid minor;

