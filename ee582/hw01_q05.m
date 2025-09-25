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

%% ===================== Require eigvals ========================
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


if ~exist('eigvals','var')
    error('Please define eigvals first (e.g., eigvals = eig(A);)');
end

%% nice colors
wineRed   = [0.60 0.00 0.10];
lushGreen = [0.00 0.55 0.20];

%% ====== stability functions ======
R_euler   = @(z) 1 + z;
R_taylor5 = @(z) 1 + z + z.^2/2 + z.^3/6 + z.^4/24 + z.^5/120;

%% ====== dense grid to draw exact |R(z)|=1 boundaries ======
[xg,yg] = meshgrid(linspace(-8, 2, 1400), linspace(-5, 5, 1400));
Z = xg + 1i*yg;

absReul = abs(R_euler(Z));
absRt5  = abs(R_taylor5(Z));

%% ====== find h_max by scan; then make 10 linear h's around it ======
[hEmax,~] = hmax_scan(R_euler,   eigvals, 'normal');
[hTmax,~] = hmax_scan(R_taylor5, eigvals, 'wide');

hEuler = linspace(0.6*hEmax, 1.4*hEmax, 10);
hT5    = linspace(0.6*hTmax,  1.6*hTmax, 10);

%% ===================== plotting =====================
fig = figure('Name','Stability regions (zoomed, green=stable / red=unstable)'); 
tlo = tiledlayout(fig,1,2,'Padding','compact','TileSpacing','compact');

% ---------- Forward Euler ----------
ax1 = nexttile(tlo); hold(ax1,'on'); box(ax1,'on'); axis(ax1,'equal');
contour(ax1, xg, yg, absReul, [1 1], 'k', 'LineWidth', 1.5);
format_axes(ax1);
title(ax1, 'Forward Euler (ode1)', 'interpreter', 'latex', 'Color','k');
subtitle(ax1, sprintf('$h \\in [%.3f,\\,%.3f]~\\mathrm{ms}$', ...
        hEuler(1)*1e3, hEuler(end)*1e3), ...
        'Interpreter','latex','FontSize',11,'Color','k');

% Restrict to ultra-relevant slice near origin
xlim(ax1, [-0.01 0]); 
ylim(ax1, [-0.035 0.035]);


for hk = hEuler
    zk = hk*eigvals;
    if all(abs(R_euler(zk)) <= 1 + 1e-12)
        plot(ax1, real(zk), imag(zk), 'x', 'Color', lushGreen);
    else
        plot(ax1, real(zk), imag(zk), 'x', 'Color', wineRed);
    end
end

% ---------- RK-5 ----------
ax2 = nexttile(tlo); hold(ax2,'on'); box(ax2,'on'); axis(ax2,'equal');
contour(ax2, xg, yg, absRt5, [1 1], 'k', 'LineWidth', 1.5);
format_axes(ax2);
% title(ax2, 'Explicit RK5 (ode5)', 'Color','k');
title(ax2, 'Explicit RK5 (ode5)', ...
         'Interpreter','latex', 'Color', 'k')
subtitle(ax2, sprintf('$h \\in [%.3f,\\,%.3f]~\\mathrm{ms}$', hT5(1)*1e3, hT5(end)*1e3), ...
         'Interpreter','latex','FontSize',11,'Color','k');

[xl2,yl2] = zoom_lobe(xg,yg,(absRt5<=1),0.10); xlim(ax2,xl2); ylim(ax2,yl2);

for hk = hT5
    zk = hk*eigvals;
    if all(abs(R_taylor5(zk)) <= 1 + 1e-12)
        plot(ax2, real(zk), imag(zk), 'x', 'Color', lushGreen);
    else
        plot(ax2, real(zk), imag(zk), 'x', 'Color', wineRed);
    end
end

fprintf('h_max guide (ms):  Euler ≈ %.3f ms | Taylor-5 ≈ %.3f ms\n', hEmax*1e3, hTmax*1e3);

%% ===================== local helpers =====================
function format_axes(ax)
    grid(ax,'on'); ax.XMinorGrid='on'; ax.YMinorGrid='on';
    xlabel(ax,'Re(z)','Interpreter','latex','Color','k');
    ylabel(ax,'Im(z)','Interpreter','latex','Color','k');
end

function [xl,yl] = zoom_lobe(X,Y,mask,padFrac)
    if ~any(mask,'all'), xl=[min(X(:)) max(X(:))]; yl=[min(Y(:)) max(Y(:))]; return; end
    xs=X(mask); ys=Y(mask); xlo=min(xs); xhi=max(xs); ylo=min(ys); yhi=max(ys);
    dx=padFrac*(xhi-xlo); dy=padFrac*(yhi-ylo);
    xl=[xlo-dx, xhi+dx]; yl=[ylo-dy, yhi+dy];
end

function [hmax, hnext] = hmax_scan(Rfun, eigs, mode)
    lam_mag = max(abs(eigs));
    if nargin<3, mode='normal'; end
    switch mode
        case 'wide'
            hscan = logspace(log10(1/(1e4*lam_mag)), log10(12/lam_mag), 1400);
        otherwise
            hscan = logspace(log10(1/(1e4*lam_mag)), log10(3/lam_mag), 1400);
    end
    stable = false(size(hscan));
    for i=1:numel(hscan)
        stable(i) = all(abs(Rfun(hscan(i)*eigs)) <= 1 + 1e-12);
    end
    idx = find(stable,1,'last');
    if isempty(idx), hmax = NaN; hnext = NaN; return; end
    hmax = hscan(idx);
    j = find(~stable(idx:min(idx+10,numel(hscan))),1,'first');
    hnext = hscan(min(idx + (isempty(j)*1 + ~isempty(j)*(j-1)), numel(hscan)));
end
