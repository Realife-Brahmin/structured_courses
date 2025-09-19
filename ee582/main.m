r_L = 1; L = 19e-3; C = 8.2e-6;
w_n = 1/sqrt(L*C)
zeta = 1/2*r_L/(sqrt(L/C))
w_d = w_n*sqrt(1-zeta^2)
lambda = (-r_L/L + sqrt((r_L/L)^2 - 4/(L*C)))/2
Qfactor = w_n*L/r_L
V_LL = 13.8e3;
V_ph = V_LL/sqrt(3);
f_fun = 60;
w_fun = 2*pi*f_fun;
X_L = w_fun*L;
X_C = 1/(w_fun*C);
X = X_L - X_C;
Z = r_L + 1j*X
I_ph = V_ph/Z
Q_kVAr = 3*abs(I_ph)^2*X*1e-3