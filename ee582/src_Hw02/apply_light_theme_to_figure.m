function apply_light_theme_to_figure(fig_handle)
% APPLY_LIGHT_THEME_TO_FIGURE Explicitly apply light theme to a specific figure
%
% This function ensures that a figure and all its axes use white background
% and black text, regardless of system defaults. Use this if setup_plot_theme()
% doesn't work consistently across different machines.
%
% Usage:
%   fig = figure();
%   apply_light_theme_to_figure(fig);
%
% Or for the current figure:
%   apply_light_theme_to_figure(gcf);

    if nargin < 1
        fig_handle = gcf;  % Use current figure if none specified
    end
    
    % Set figure properties
    set(fig_handle, 'Color', 'w');
    set(fig_handle, 'InvertHardcopy', 'off');
    
    % Find all axes in the figure
    all_axes = findall(fig_handle, 'Type', 'axes');
    
    for ax = all_axes'
        % Set axes background and colors
        set(ax, 'Color', 'w');
        set(ax, 'XColor', 'k');
        set(ax, 'YColor', 'k');
        
        if isprop(ax, 'ZColor')
            set(ax, 'ZColor', 'k');
        end
        
        % Grid settings
        set(ax, 'GridColor', [.2 .2 .2]);
        set(ax, 'MinorGridColor', [.6 .6 .6]);
        set(ax, 'GridAlpha', 0.3);
        set(ax, 'MinorGridAlpha', 0.15);
        
        % Text color
        if ~isempty(ax.Title)
            set(ax.Title, 'Color', 'k');
        end
        if ~isempty(ax.XLabel)
            set(ax.XLabel, 'Color', 'k');
        end
        if ~isempty(ax.YLabel)
            set(ax.YLabel, 'Color', 'k');
        end
    end
    
    % Find all legends
    all_legends = findall(fig_handle, 'Type', 'Legend');
    for leg = all_legends'
        set(leg, 'Color', 'w');
        set(leg, 'TextColor', 'k');
        set(leg, 'EdgeColor', 'k');
    end
    
    % Find all text objects
    all_text = findall(fig_handle, 'Type', 'text');
    for txt = all_text'
        if isprop(txt, 'Color')
            set(txt, 'Color', 'k');
        end
    end
    
end
