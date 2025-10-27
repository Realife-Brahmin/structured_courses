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

    %% ===================== Reset to factory defaults first =====================
    % This ensures a clean slate, overriding any system-specific settings
    set(groot, 'defaultFigureColor', 'remove');
    set(groot, 'defaultAxesColor', 'remove');
    
    %% ===================== Global light theme (EXPLICIT) =====================
    % Figure settings
    set(groot, 'defaultFigureColor', 'w');
    set(groot, 'defaultFigureInvertHardcopy', 'off');  % Prevent color inversion on save
    
    % Axes background and colors
    set(groot, 'defaultAxesColor', 'w');
    set(groot, 'defaultAxesXColor', 'k');
    set(groot, 'defaultAxesYColor', 'k');
    set(groot, 'defaultAxesZColor', 'k');
    
    % Grid settings
    set(groot, 'defaultAxesGridColor', [.2 .2 .2]);
    set(groot, 'defaultAxesMinorGridColor', [.6 .6 .6]);
    set(groot, 'defaultAxesGridAlpha', 0.3);
    set(groot, 'defaultAxesMinorGridAlpha', 0.15);
    set(groot, 'defaultAxesXMinorGrid', 'on');
    set(groot, 'defaultAxesYMinorGrid', 'on');
    
    % Font settings
    set(groot, 'defaultAxesFontName', 'Helvetica');
    set(groot, 'defaultAxesFontSize', 12);
    set(groot, 'defaultAxesTitleFontWeight', 'bold');
    set(groot, 'defaultAxesTitleFontSizeMultiplier', 1.1);
    
    % Line and marker settings
    set(groot, 'defaultLineLineWidth', 2.2);
    set(groot, 'defaultLineMarkerSize', 8);
    
    % Legend settings
    set(groot, 'defaultLegendColor', 'w');
    set(groot, 'defaultLegendTextColor', 'k');
    set(groot, 'defaultLegendEdgeColor', 'k');
    set(groot, 'defaultLegendBox', 'on');

    % Text interpreter settings (LaTeX)
    set(groot, 'defaultTextInterpreter', 'latex');
    set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'defaultLegendInterpreter', 'latex');
    set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');
    
    % Additional safeguards
    set(groot, 'defaultFigureGraphicsSmoothing', 'on');
    set(groot, 'defaultAxesBox', 'off');
    
    fprintf('[Theme] Light theme configured: White background, black text\n');

end
