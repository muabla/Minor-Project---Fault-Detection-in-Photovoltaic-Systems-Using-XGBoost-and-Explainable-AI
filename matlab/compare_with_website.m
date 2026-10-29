function compare_with_website()
%COMPARE_WITH_WEBSITE  Side-by-side: the website's reading (Python simulator) vs the last pv_check run (Simulink/Simscape).
%   Steps: 1) on the website press "Copy website values" and paste into the Command Window (creates the variable "web")
%          2) run pv_check(...) for the same fault (button "Copy MATLAB command")
%          3) run compare_with_website
if ~evalin('base', 'exist(''web'', ''var'')')
    error('No "web" variable. On the website press "Copy website values" and paste it into the Command Window first.');
end
if ~evalin('base', 'exist(''pvlast'', ''var'')')
    error('No Simulink result yet. Run pv_check(...) first (website button "Copy MATLAB command").');
end
web = evalin('base', 'web'); sim = evalin('base', 'pvlast'); info = pv_get('pv_info');
if web.system ~= sim.system || ~strcmp(web.config, sim.config)
    warning('The website reading is System %d (%s) but the last pv_check run is System %d (%s).', ...
        web.system, web.config, sim.system, sim.config);
end

fprintf('\n=== Website (Python simulator)  vs  Simulink (Simscape circuit) ===   System %d, %s\n\n', sim.system, sim.config);
fprintf('%-24s %14s %14s %10s\n', '', 'Website', 'Simulink', 'diff');
q = {'Voc (V)', web.Voc, sim.Voc; 'Isc (A)', web.Isc, sim.Isc; 'Vmp (V)', web.Vmp, sim.Vmp; ...
     'Imp (A)', web.Imp, sim.Imp; 'Max power (W)', web.P, sim.Vmp * sim.Imp};
for k = 1:size(q, 1)
    fprintf('%-24s %14.3f %14.3f %9.2f%%\n', q{k, 1}, q{k, 2}, q{k, 3}, pct(q{k, 3}, q{k, 2}));
end
fprintf('%-24s %14s %14s\n', 'Irradiance used (W/m2)', num2str(web.G), num2str(sim.G));

fprintf('\n%-24s %14s %14s\n', 'PREDICTION', upper(web.pred), upper(sim.label));
fprintf('%-24s %14s %14s\n', 'Probabilities:', 'Website', 'Simulink');
for k = 1:6
    fprintf('  %-22s %13.1f%% %13.1f%%\n', info.classes{k}, 100 * web.probs(k), 100 * sim.probs(k));
end
if strcmp(web.pred, sim.label)
    fprintf('\nSAME DIAGNOSIS in both simulators.\n');
else
    fprintf('\nDIFFERENT DIAGNOSIS - send this output to Claude.\n');
end

fprintf('\nThe 22 model inputs:\n%-12s %10s %10s %9s\n', '', 'Website', 'Simulink', 'diff');
for k = 1:22
    fprintf('%-12s %10.4f %10.4f %8.2f%%\n', info.feature_order{k}, web.inputs(k), sim.inputs(k), pct(sim.inputs(k), web.inputs(k)));
end
fprintf('(Small differences are expected: the website adds sensor noise; Simulink is noise-free.)\n\n');

figure('Name', 'Website vs Simulink I-V curve', 'Color', 'w');
plot(web.V, web.I, 'b-', 'LineWidth', 1.5); hold on
plot(sim.V, sim.I, 'r--', 'LineWidth', 1.5);
grid on; xlabel('Voltage (V)'); ylabel('Current (A)');
legend(sprintf('Website (Python): %s', web.pred), sprintf('Simulink (Simscape): %s', sim.label), 'Location', 'southwest');
title(sprintf('System %d - same fault simulated by both', sim.system));
end

function d = pct(a, b)
if abs(b) > 1e-9
    d = (a - b) / abs(b) * 100;
else
    d = 0;
end
end
