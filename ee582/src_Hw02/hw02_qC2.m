% hw02_qC2.m
% EE582 HW02 - Section C.2: Circuit Interruption Simulation
% Author: Aryan Ritwajeet Jha
% Date: October 2025
%
% Purpose: 
%   Implement MATLAB nodal analysis solution for the transmission line
%   breaker interruption problem. Simulate the circuit from t=0 to t=20ms
%   with breaker opening at t=8ms (when current crosses zero).
%
% Uses the recommended time-step h = 35 μs from Question C.1

clear; clc; close all;

% Add path to helper functions
addpath('../src_Hw02/');  % Access setup_plot_theme and other utilities

setup_plot_theme(); % Consistent plot style

%% ========================================================================
%  SECTION 1: CIRCUIT PARAMETERS
%  ========================================================================
fprintf('\n=================================================================\n');
fprintf('  EE582 HW02 - Circuit Interruption Simulation (Question C.2)\n');
fprintf('=================================================================\n\n');

% Circuit parameters (same as Question C.1)
% Source voltage
V_peak = 230e3;         % Peak voltage [V] (230 kV)
omega = 377;            % Angular frequency [rad/s] (60 Hz)

% Left side (supply system + transformer + substation capacitance)
R1 = 3;                 % Resistance [Ohm]
L1 = 350e-3;            % Inductance [H]
C1 = 10e-9;             % Capacitance [F]

% Right side (transmission line parameters)
R2 = 50;                % Resistance [Ohm]
L2 = 100e-3;            % Inductance [H]
C2 = 600e-9;            % Capacitance [F]

% Simulation parameters
h = 35e-6;              % Time-step [s] (35 μs, from C.1 analysis)
T_start = 0;            % Start time [s]
T_end = 20e-3;          % End time [s] (20 ms)
t_breaker_cmd = 8e-3;   % Breaker command time [s] (8 ms)

% Time vector
t = T_start:h:T_end;
N = length(t);

fprintf('CIRCUIT PARAMETERS:\n');
fprintf('-------------------\n');
fprintf('Source: V_peak = %.0f kV, ω = %.0f rad/s (f = %.1f Hz)\n', ...
    V_peak/1e3, omega, omega/(2*pi));
fprintf('Left:   R1 = %.2f Ω, L1 = %.0f mH, C1 = %.0f nF\n', R1, L1*1e3, C1*1e9);
fprintf('Right:  R2 = %.2f Ω, L2 = %.0f mH, C2 = %.0f nF\n\n', R2, L2*1e3, C2*1e9);

fprintf('SIMULATION PARAMETERS:\n');
fprintf('----------------------\n');
fprintf('Time-step:       h = %.2f μs\n', h*1e6);
fprintf('Simulation time: T = %.1f ms (from %.1f to %.1f ms)\n', ...
    (T_end-T_start)*1e3, T_start*1e3, T_end*1e3);
fprintf('Breaker command: t = %.1f ms\n', t_breaker_cmd*1e3);
fprintf('Number of steps: N = %d\n\n', N);

%% ========================================================================
%  SECTION 2: COMPANION MODEL PARAMETERS (TRAPEZOIDAL DISCRETIZATION)
%  ========================================================================
fprintf('COMPANION MODEL PARAMETERS:\n');
fprintf('---------------------------\n');

% Inductor companion models (Trapezoidal)
R_L1 = 2*L1/h;          % Equivalent resistance for L1 [Ohm]
R_L2 = 2*L2/h;          % Equivalent resistance for L2 [Ohm]

% Capacitor companion models (Trapezoidal)
R_C1 = h/(2*C1);        % Equivalent resistance for C1 [Ohm]
R_C2 = h/(2*C2);        % Equivalent resistance for C2 [Ohm]

fprintf('Inductors:\n');
fprintf('  L1: R_L1 = %.4e Ω\n', R_L1);
fprintf('  L2: R_L2 = %.4e Ω\n', R_L2);
fprintf('Capacitors:\n');
fprintf('  C1: R_C1 = %.4e Ω\n', R_C1);
fprintf('  C2: R_C2 = %.4e Ω\n\n', R_C2);

%% ========================================================================
%  SECTION 3: INITIALIZE STATE VARIABLES
%  ========================================================================
fprintf('Initializing state variables...\n');

% Node voltages (assuming 4 nodes based on circuit)
% Node 1: After source/R1
% Node 2: Left of breaker (after C1)
% Node 3: Right of breaker (after C2)
% Node 4: Ground reference (implicit, not stored)

% TODO: Define node voltage arrays
% v1 = zeros(1, N);  % Node 1 voltage
% v2 = zeros(1, N);  % Node 2 voltage
% v3 = zeros(1, N);  % Node 3 voltage

% TODO: Define current arrays for components
% i_L1 = zeros(1, N);  % Inductor L1 current
% i_L2 = zeros(1, N);  % Inductor L2 current
% i_C1 = zeros(1, N);  % Capacitor C1 current
% i_C2 = zeros(1, N);  % Capacitor C2 current

% TODO: Define history terms for companion models
% e_h_L1 = 0;  % History voltage for L1
% e_h_L2 = 0;  % History voltage for L2
% e_h_C1 = 0;  % History voltage for C1
% e_h_C2 = 0;  % History voltage for C2

% Breaker state
breaker_closed = true;
breaker_open_time = NaN;  % Will be set when breaker actually opens

fprintf('State variables initialized.\n\n');

%% ========================================================================
%  SECTION 4: TIME-STEPPING LOOP
%  ========================================================================
fprintf('Starting time-stepping simulation...\n');
fprintf('==================================\n\n');

for n = 1:N
    % Current time
    t_now = t(n);
    
    % Source voltage at current time
    e_t = V_peak * cos(omega * t_now);
    
    % TODO: Check if breaker should open
    % Breaker opens when:
    %   1. Command has been issued (t >= t_breaker_cmd)
    %   2. Current through breaker crosses zero
    
    % TODO: Build conductance matrix G and current vector I
    % based on breaker state (closed or open)
    
    % TODO: Solve nodal equations: G * v = I
    
    % TODO: Update history terms for next time step
    
    % Progress indicator (every 1000 steps)
    if mod(n, 1000) == 0
        fprintf('  Step %d/%d (t = %.3f ms)\n', n, N, t_now*1e3);
    end
end

fprintf('\nSimulation complete!\n');
if ~isnan(breaker_open_time)
    fprintf('Breaker opened at t = %.4f ms\n', breaker_open_time*1e3);
else
    fprintf('Warning: Breaker did not open during simulation.\n');
end
fprintf('\n');

%% ========================================================================
%  SECTION 5: VISUALIZATION
%  ========================================================================
fprintf('Generating plots...\n');

% TODO: Create plots for:
%   - Node voltages (v1, v2, v3)
%   - Component currents
%   - Breaker current
%   - Fault current

% Save figures
figuresFolder = "../tex_Hw02/figures/";
if ~exist(figuresFolder, 'dir'), mkdir(figuresFolder); end

fprintf('Figures will be saved to: %s\n', figuresFolder);

fprintf('\n=================================================================\n');
fprintf('  Analysis Complete!\n');
fprintf('=================================================================\n');
