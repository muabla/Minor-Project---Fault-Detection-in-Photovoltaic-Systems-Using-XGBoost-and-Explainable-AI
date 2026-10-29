function check_healthy(system, config, full)
%CHECK_HEALTHY  Calibration check: the HEALTHY array simulated by Simscape vs the Python simulator
%   (Voc, Isc, Vmp, Imp, maximum power) at several sunlight / temperature points.
%   check_healthy(3)  check_healthy(2)  check_healthy(1,'series')  check_healthy(1,'parallel')
%   check_healthy(3,'series',true)  -> all 12 points instead of 4
if nargin < 2, config = 'series'; end
if nargin < 3, full = false; end
here = fileparts(mfilename('fullpath'));
T = readtable(fullfile(here, 'pv_data', 'healthy_reference.csv'), 'TextType', 'string');
T = T(T.system == system & T.config == config, :);
if ~full
    pick = (T.G == 1000 & T.Ta == 25) | (T.G == 800 & T.Ta == 25) | (T.G == 500 & T.Ta == 15) | (T.G == 250 & T.Ta == 40);
    T = T(pick, :);
end
fprintf('\n=== check_healthy: System %d (%s), Simscape vs Python, healthy array ===\n', system, config);
fprintf('%6s %5s %6s | %-24s | %-24s | %-24s | %-24s | %s\n', 'G', 'Ta', 'Tc', 'Voc (V)  py / sim', ...
    'Isc (A)  py / sim', 'Vmp (V)  py / sim', 'Pmax (W) py / sim', 'worst');
worst = 0;
for k = 1:height(T)
    s = struct('G', T.G(k), 'Ta', T.Ta(k), 'Tc', T.Tc(k), 'light', [], 'short', zeros(0, 3), 'open', 0, 'corrosion', 0, 'rbranch', 1);
    r = pv_simulate(system, config, s, zeros(1, 6));
    d = [pct(r.Voc, T.Voc(k)), pct(r.Isc, T.Isc(k)), pct(r.Vmp, T.Vmp(k)), pct(r.Vmp * r.Imp, T.Pmp(k))];
    worst = max(worst, max(abs(d)));
    fprintf('%6g %5g %6.1f | %8.2f / %8.2f %5.1f%% | %8.3f / %8.3f %5.1f%% | %8.2f / %8.2f %5.1f%% | %8.1f / %8.1f %5.1f%% | %.1f%%\n', ...
        T.G(k), T.Ta(k), T.Tc(k), T.Voc(k), r.Voc, d(1), T.Isc(k), r.Isc, d(2), T.Vmp(k), r.Vmp, d(3), T.Pmp(k), r.Vmp * r.Imp, d(4), max(abs(d)));
end
if worst <= 2
    fprintf('PASS: Simscape and Python agree within %.1f %% (limit 2 %%).\n\n', worst);
else
    fprintf('NOT YET: largest difference %.1f %% (limit 2 %%). Send this output to Claude.\n\n', worst);
end
end

function d = pct(a, b)
d = (a - b) / max(abs(b), 1e-9) * 100;
end
