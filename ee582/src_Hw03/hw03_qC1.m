% hw03_qC1.m
% EE582 HW03 - Section C.1: Resonant Frequency and Nyquist Sampling Analysis
% Author: Aryan Ritwajeet Jha
% Date: October 2025
%
% Purpose: 
%   Compute resonant frequencies for different circuit states and determine
%   recommended time-steps for trapezoidal discretization based on Nyquist
%   sampling criteria.
%
% Circuit States Analyzed:
%   1. Closed state (full circuit)
%   2. Open state - Left portion
%   3. Open state - Right portion
%
% Formulas:
%   - Resonant frequency: f_res = 1/(2*pi*sqrt(L*C)) for LC circuit
%   - Maximum frequency: f_max = f_res (for oscillatory response)
%   - Recommended time-step (3% error): h = 1/(10*f_max)

clear; clc; close all;

% Add path to helper functions (if needed)
addpath('../src_Hw02/');  % Access setup_plot_theme and other utilities

setup_plot_theme(); % Consistent plot style

%% ========================================================================
%  SECTION 1: CIRCUIT PARAMETERS
%  ========================================================================
fprintf('\n=================================================================\n');
fprintf('  EE582 HW03 - Resonant Frequency & Time-step Analysis\n');
fprintf('=================================================================\n\n');

% TODO: Replace with actual circuit parameter values


% Circuit parameters - OPEN STATE: LEFT PORTION
R1 = 3;            % Resistance [Ohm] - UPDATE THIS
L1 = 350e-3;          % Inductance [H] - UPDATE THIS
C1 = 10e-9;         % Capacitance [F] - UPDATE THIS

% Circuit parameters - OPEN STATE: RIGHT PORTION
R2 = 50;           % Resistance [Ohm] - UPDATE THIS
L2 = 100e-3;         % Inductance [H] - UPDATE THIS
C2 = 600e-9;        % Capacitance [F] - UPDATE THIS

% Circuit parameters - CLOSED STATE (full circuit)
L = (L1*L2)/(L1+L2);        % Inductance [H] - UPDATE THIS
C = C1+C2;      % Capacitance [F] - UPDATE THIS

% Display circuit parameters
fprintf('CIRCUIT PARAMETERS:\n');
fprintf('-------------------\n');
fprintf('Closed State:  L = %.4f H, C = %.2e F\n', L, C);
fprintf('Open (Left):   R = %.2f Ω, L = %.4f H, C = %.2e F\n', R1, L1, C1);
fprintf('Open (Right):  R = %.2f Ω, L = %.4f H, C = %.2e F\n\n', R2, L2, C2);

%% ========================================================================
%  SECTION 2: RESONANT FREQUENCY CALCULATIONS
%  ========================================================================
fprintf('RESONANT FREQUENCY CALCULATIONS:\n');
fprintf('--------------------------------\n');

% Closed state resonant frequency
% For series RLC: f_res = 1/(2*pi*sqrt(L*C))
f_res_closed = 1 / (2 * pi * sqrt(L * C));
omega_res_closed = 2 * pi * f_res_closed;

fprintf('Closed State:\n');
fprintf('  f_res = %.4f Hz (%.4f kHz)\n', f_res_closed, f_res_closed/1e3);
fprintf('  ω_res = %.4f rad/s\n', omega_res_closed);
fprintf('  T_res = %.6f s (%.4f ms)\n\n', 1/f_res_closed, 1e3/f_res_closed);

% Open state - Left portion
f_res_left = 1 / (2 * pi * sqrt(L1 * C1));
omega_res_left = 2 * pi * f_res_left;

fprintf('Open State (Left Portion):\n');
fprintf('  f_res = %.4f Hz (%.4f kHz)\n', f_res_left, f_res_left/1e3);
fprintf('  ω_res = %.4f rad/s\n', omega_res_left);
fprintf('  T_res = %.6f s (%.4f ms)\n\n', 1/f_res_left, 1e3/f_res_left);

% Open state - Right portion
f_res_right = 1 / (2 * pi * sqrt(L2 * C2));
omega_res_right = 2 * pi * f_res_right;

fprintf('Open State (Right Portion):\n');
fprintf('  f_res = %.4f Hz (%.4f kHz)\n', f_res_right, f_res_right/1e3);
fprintf('  ω_res = %.4f rad/s\n', omega_res_right);
fprintf('  T_res = %.6f s (%.4f ms)\n\n', 1/f_res_right, 1e3/f_res_right);

%% ========================================================================
%  SECTION 3: RECOMMENDED TIME-STEP FOR TRAPEZOIDAL DISCRETIZATION
%  ========================================================================
fprintf('RECOMMENDED TIME-STEP CALCULATION:\n');
fprintf('----------------------------------\n');
fprintf('Formula: h = 1 / (10 * f_max) for 3%% error with Trapezoidal rule\n');
fprintf('where f_max = f_res for each circuit state\n\n');

% Maximum frequency = resonant frequency (for oscillatory response)
f_max_closed = f_res_closed;
f_max_left = f_res_left;
f_max_right = f_res_right;

% Recommended time-step: h = 1/(10*f_max) for 3% error
h_closed = 1 / (10 * f_max_closed);
h_left = 1 / (10 * f_max_left);
h_right = 1 / (10 * f_max_right);

fprintf('Closed State:  h = %.4e s (%.4f μs)\n', h_closed, h_closed*1e6);
fprintf('Open (Left):   h = %.4e s (%.4f μs)\n', h_left, h_left*1e6);
fprintf('Open (Right):  h = %.4e s (%.4f μs)\n\n', h_right, h_right*1e6);

% Overall recommendation (most conservative - minimum value)
h_recommended = min([h_closed, h_left, h_right]);

fprintf('OVERALL RECOMMENDATION:\n');
fprintf('=======================\n');
fprintf('Use time-step: h = %.4e s (%.4f μs)\n', h_recommended, h_recommended*1e6);
fprintf('This is the minimum value to handle all circuit transients.\n\n');

fprintf('=================================================================\n');
fprintf('  Analysis Complete!\n');
fprintf('=================================================================\n');