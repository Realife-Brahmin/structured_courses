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

r_L = 1; L = 19e-3; C_cap = 8.2e-6;
w_n = 1/sqrt(L*C_cap)
zeta = 1/2*r_L/(sqrt(L/C_cap))
w_d = w_n*sqrt(1-zeta^2)
lambda = (-r_L/L + sqrt((r_L/L)^2 - 4/(L*C_cap)))/2
Qfactor = w_n*L/r_L
V_LL = 13.8e3;
V_ph = V_LL/sqrt(3);
f_fun = 60;
w_fun = 2*pi*f_fun;
X_L = w_fun*L;
X_C = 1/(w_fun*C_cap);
X = X_L - X_C;
Z = r_L + 1j*X
I_ph = V_ph/Z
Q_kVAr = 3*abs(I_ph)^2*X*1e-3
phi = acos(zeta)
t = 0:0.001:0.2;
x_ss_kV = 11.3;
x_t_kV = x_ss_kV * (1 - (1/sqrt(1-zeta^2)) * exp(-zeta*w_n*t) .* sin(w_d*t + phi));
plot(t, x_t_kV)
xlabel('Time [s]')
ylabel('Voltage [kV]')
