figure('Color', 'w');

subplot(2,1,1);
ax1 = gca;
plot(out.i_L_t.Time, out.i_L_t.Data * 1000, 'b', 'LineWidth', 1.5);
ylabel('i_L(t) [A]', 'Color', 'k');
title('Inductor Current', 'Color', 'k');
set(ax1, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');
grid on;
ax1.MinorGridLineStyle = '-';
ax1.MinorGridAlpha = 0.18; % moderate transparency
ax1.MinorGridColor = [0.5 0.5 0.5]; % medium gray
grid minor;

subplot(2,1,2);
ax2 = gca;
plot(out.V_c_t.Time, out.V_c_t.Data, 'r', 'LineWidth', 1.5);
ylabel('V_C(t) [kV]', 'Color', 'k');
xlabel('Time (s)', 'Color', 'k');
title('Capacitor Voltage', 'Color', 'k');
set(ax2, 'XColor', 'k', 'YColor', 'k', 'Color', 'w');
grid on;
ax2.MinorGridLineStyle = '-';
ax2.MinorGridAlpha = 0.18; % moderate transparency
ax2.MinorGridColor = [0.5 0.5 0.5]; % medium gray
grid minor;