function setup_plot_theme()
% SETUP_PLOT_THEME Configure global plot theme with light background and LaTeX fonts
%
% This function sets up a consistent plotting theme across all scripts with:
% - Light background (white)
% - Black text and axes
% - LaTeX interpreter for all text elements
% - Minor grids enabled
% - Professional font settings
%
% Usage: Call this function at the beginning of any plotting script
%   setup_plot_theme();

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
    set(groot,'defaultLineLineWidth',2.2);
    set(groot,'defaultLineMarkerSize',8);
    set(groot, 'defaultLegendColor', 'w');      % White background for all legends
    set(groot, 'defaultLegendTextColor', 'k');  % Black text for all legends

    % LaTeX interpreter for all text
    set(groot,'defaultTextInterpreter','latex');
    set(groot,'defaultAxesTickLabelInterpreter','latex');
    set(groot,'defaultLegendInterpreter','latex');

end
