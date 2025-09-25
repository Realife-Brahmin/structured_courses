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

thisFolder = fileparts(which(mfilename));
addpath(genpath(thisFolder));

rawDataFolder = "./rawData/";

%% ----------------------- GIVEN (your variables) -----------------------
r_L = 1; L = 19e-3; C_cap = 8.2e-6;
r_L2 = 323e-3; r_L1 = r_L - r_L2;
L_2 = 1e-3;   L_1 = L - L_2;
r_C = 1e6;
r_leak = 30e3;

V_LL = 13.8e3;
V_ph = V_LL/sqrt(3);
f_fun = 60;
w_fun = 2*pi*f_fun;

T_horizon_s = 0.2;
h_ms = 0.1;
h_s = h_ms*1e-3;
t = 0:h_s:T_horizon_s;

%% =====================================================================
%                       PART A (simple series RLC)
%                   (kept here for cross-check/reference)
%% =====================================================================
w_n_A   = 1/sqrt(L*C_cap);
zeta_A  = 0.5 * r_L * sqrt(C_cap/L);
w_d_A   = w_n_A*sqrt(max(0,1-zeta_A^2));         % handle tiny num error
lambdaA = (-r_L/L + sqrt((r_L/L)^2 - 4/(L*C_cap)))/2;
Qfactor = w_n_A*L/r_L;

% Phasor vars at 60 Hz (exact)
X_L = w_fun*L;               % inductive reactance
X_C = 1/(w_fun*C_cap);       % capacitive reactance
X   = X_L - X_C;             % net series reactance
Z   = r_L + 1i*X;
I_ph = V_ph/Z;
Q_kVAr = 3*abs(I_ph)^2*X*1e-3;   % +inductive / -capacitive

%% =====================================================================
%               PART B (disconnected case, Fig. 1b)
%      Overdamped discharge of vC(t) with rL2, L2, rC, r_leak
%% =====================================================================
rSigma = r_L2 + r_leak;                % r_Σ = r_L2 + r_leak

% Monic ODE: vC'' + a1*vC' + a0*vC = 0
a1 = rSigma/L_2 + 1/(r_C*C_cap);
a0 = (1 + rSigma/r_C)/(L_2*C_cap);

% Second-order canonical parameters
w_n   = sqrt(a0);
zeta  = a1/(2*w_n);
% Overdamped -> no oscillation; report wd=NaN
w_d   = NaN;

% Poles and time constants
p = roots([1 a1 a0]);                % s1, s2 (both < 0)
[~,idx] = sort(abs(p),'descend');    % p(idx(1)) = fast (larger |s|)
s_fast = p(idx(1));  s_slow = p(idx(2));
tau_fast = -1/real(s_fast);
tau_slow = -1/real(s_slow);

% Practical discharge times
t_4tau = 4*tau_slow;     % ~98.2% discharged
t_5tau = 5*tau_slow;     % ~99.3% discharged

%% vC(t) for the disconnected case (overdamped homogeneous solution)
% Initial conditions: assume filter is opened with capacitor at V0 and
% inductor current zero at the instant of disconnect (i_L2(0)=0).
V0_kV = 11.3;                   % initial capacitor voltage in kV (your A2 step height)
V0    = V0_kV*1e3;              % volts

% i_L2 = C*vC' + vC/rC.  i_L2(0)=0 -> vC'(0) = -V0/(rC*C)
vC0  = V0;
vCdot0 = -V0/(r_C*C_cap);

% vC(t) = c1*e^{s1 t} + c2*e^{s2 t}, with:
s1 = p(1); s2 = p(2);
c1 = (vCdot0 - s2*vC0)/(s1 - s2);
c2 = vC0 - c1;

vC_t = c1*exp(s1*t) + c2*exp(s2*t);     % volts
iL2_t = C_cap*gradient(vC_t,h_s) + vC_t/r_C;  % A (from KCL), optional

% If you want the Part-A *underdamped* step closed-form for comparison:
phiA = atan2(sqrt(max(0,1-zeta_A^2)), zeta_A);
x_ss_kV = 11.3;
x_t_kV_A = x_ss_kV * (1 - (1/sqrt(max(1e-16,1-zeta_A^2))) ...
            .* exp(-zeta_A*w_n_A*t) .* sin(w_d_A*t + phiA));

% For the disconnected **overdamped** case, plot vC in kV:
x_t_kV = vC_t/1e3;

%% --------------------------- PRINT SUMMARY ---------------------------
fprintf('\n=== Part A (simple RLC) ===\n');
fprintf('w_n_A = %.6g rad/s,  zeta_A = %.6g,  w_d_A = %.6g rad/s\n', w_n_A, zeta_A, w_d_A);
fprintf('Q = %.6g,  lambda (one root) = %.6g 1/s\n', Qfactor, lambdaA);
fprintf('Q_total@60Hz = %.3f kVAr (sign from X)\n', Q_kVAr);

fprintf('\n=== Part B (disconnected, overdamped) ===\n');
fprintf('a1 = %.9g 1/s,  a0 = %.9g 1/s^2\n', a1, a0);
fprintf('w_n = %.6g rad/s,  zeta = %.6g,  w_d = NaN (overdamped)\n', w_n, zeta);
fprintf('poles s1=%.9g 1/s, s2=%.9g 1/s\n', s1, s2);
fprintf('tau_fast = %.3e s (%.3f ms)\n', tau_fast, 1e3*tau_fast);
fprintf('tau_slow = %.6f s (%.3f ms)\n', tau_slow, 1e3*tau_slow);
fprintf('~98%% discharge (4 tau) = %.3f s,   ~99.3%% (5 tau) = %.3f s\n', t_4tau, t_5tau);

%% ----------------------------- PLOTS ---------------------------------
figure('Name','Disconnected capacitor discharge (overdamped)');
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

nexttile;
plot(t, x_t_kV, 'LineWidth',1.4); grid on;
ylabel('v_C(t) [kV]'); title('Capacitor voltage (disconnected, overdamped)');

nexttile;
plot(t, iL2_t, 'LineWidth',1.2); grid on;
ylabel('i_{L2}(t) [A]'); xlabel('Time (s)'); title('Series path current');

