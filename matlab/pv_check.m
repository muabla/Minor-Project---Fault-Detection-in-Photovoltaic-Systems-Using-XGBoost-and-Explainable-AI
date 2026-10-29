function pv_check(system, varargin)
%PV_CHECK  Re-simulates a fault in Simulink/Simscape and lets the Raspberry Pi subsystem (XGBoost) diagnose it.
%   Normally pasted from the website (button "Copy MATLAB command"), e.g.
%     pv_check(3, 'G',800, 'Ta',25, 'Tc',50.5, 'config','series', 'short',[1 2 1; 1 4 3])
%   Options: G (W/m2), Ta (C), Tc (cell temperature, C; default Ta + heating), config ('series'/'parallel', System 1),
%            light (modules x cells, fraction of G), short ([string module substring], substring 0 = whole module),
%            open (string number), corrosion (fraction of Vmp/Imp), rbranch (string), model ('zero' or 'adapted')
%   Two runs: (1) the same array healthy in the same weather -> "expected healthy" values for the ratio inputs,
%             (2) the array with the fault -> the diagnosis.
p = inputParser;
p.addParameter('G', 800); p.addParameter('Ta', 25); p.addParameter('Tc', []);
p.addParameter('config', 'series'); p.addParameter('light', []); p.addParameter('short', zeros(0, 3));
p.addParameter('open', 0); p.addParameter('corrosion', 0); p.addParameter('rbranch', 1);
p.addParameter('model', 'zero');
p.parse(varargin{:}); o = p.Results;
info = pv_get('pv_info');
heat = info.systems.(sprintf('s%d', system)).heat;
TcNom = o.Ta + heat * o.G;
if isempty(o.Tc), o.Tc = TcNom; end
if strcmp(o.model, 'adapted')
    A = pv_get('xgbAdapted'); assignin('base', 'xgbMdl', A{system}); mname = 'after 5 readings per class';
else
    assignin('base', 'xgbMdl', pv_get('xgbZero')); mname = 'library model (zero-shot)';
end
fprintf('\n=== pv_check: System %d (%s), G = %g W/m2, Ta = %g C, cell %.1f C ===\n', system, o.config, o.G, o.Ta, o.Tc);

% (1) healthy reference, same weather (cached)
persistent cache
if isempty(cache), cache = containers.Map(); end
key = sprintf('%d_%s_%.1f_%.2f', system, o.config, o.G, o.Ta);
if ~isKey(cache, key)
    fprintf('Run 1/2: healthy array, same weather (expected values)...\n');
    h = pv_simulate(system, o.config, struct('G', o.G, 'Ta', o.Ta, 'Tc', TcNom, 'light', [], 'short', zeros(0, 3), ...
        'open', 0, 'corrosion', 0, 'rbranch', 1), zeros(1, 6));
    h.ex = [h.x(1) h.x(2) h.x(5) h.x(6) h.x(7) h.x(12)];
    cache(key) = h;
else
    fprintf('Run 1/2: healthy reference already computed for this weather.\n');
end
h = cache(key);

% (2) the fault
fprintf('Run 2/2: array with the fault...\n');
scen = struct('G', o.G, 'Ta', o.Ta, 'Tc', o.Tc, 'light', o.light, 'short', o.short, 'open', o.open, ...
    'corrosion', o.corrosion, 'rbranch', o.rbranch);
r = pv_simulate(system, o.config, scen, h.ex, true);
names = info.classes;
fprintf('\nModel: %s\nDIAGNOSIS (Raspberry Pi subsystem, XGBoost): %s\n\n', mname, upper(names{r.labelIdx}));
for k = 1:6
    fprintf('  %-16s %6.1f %%  %s\n', names{k}, 100 * r.probs(k), repmat('#', 1, round(30 * r.probs(k))));
end
fprintf('\n  %-14s %10s %10s\n', '', 'healthy', 'this fault');
fprintf('  %-14s %10.2f %10.2f\n', 'Voc (V)', h.Voc, r.Voc, 'Isc (A)', h.Isc, r.Isc, 'Vmp (V)', h.Vmp, r.Vmp, ...
    'Imp (A)', h.Imp, r.Imp, 'Power (W)', h.Vmp * h.Imp, r.Vmp * r.Imp);
fprintf('  (simulation time %.0f s)\n', r.seconds);

% diagnostic: the 22 inputs computed by the Pi block, and predict() on exactly these inputs (outside Simulink)
mdlNow = pv_get('xgbMdl');
[lp, sp] = predict(mdlNow, r.x);
fprintf('\nCHECK: MATLAB predict() on the same 22 inputs -> %s (%.1f %%); Simulink XGBoost block -> %s (%.1f %%)\n', ...
    names{pv_label_index(lp, mdlNow)}, 100 * max(sp), names{r.labelIdx}, 100 * max(r.probs));
fprintf('The 22 inputs computed by the Raspberry Pi block:\n');
for k = 1:22
    fprintf('  %-11s %10.5f\n', info.feature_order{k}, r.x(k));
end

% SHAP explanation computed inside the Raspberry Pi subsystem
fprintf('\nWHY? (SHAP, inside the Raspberry Pi subsystem) - evidence for "%s":\n', names{r.labelIdx});
if any(r.shap)
    R = pv_reasons(r.x, h.x, r.shap, 6);
    fprintf('  %s\n', R{:});
    % main reason: the strongest supporting input that is actually abnormal (a "normal" input only rules others out)
    Rall = pv_reasons(r.x, h.x, r.shap, 22); main = '';
    for q = 1:numel(Rall)
        if contains(Rall(q), '[supports]') && ~contains(Rall(q), 'is normal') && ~contains(Rall(q), 'Weather input') ...
                && ~contains(Rall(q), 'parallel string'), main = Rall(q); break, end
    end
    if strlength(main) == 0, main = Rall(1); end
    fprintf('  Main reason: %s\n', regexprep(main, '^\S+\s+\[[^\]]*\]\s+', ''));
else
    fprintf('  (no SHAP values - the SHAP block did not run)\n');
end
% protection
if r.trip > 0
    fprintf('\nPROTECTION: short circuit -> the Pi opened DC contactor K1 (0.1 s relay delay). Array current after the trip: %.3f A\n\n', r.Iall(end));
else
    fprintf('\nPROTECTION: no trip - DC contactor K1 stays closed (current at the end: %.2f A).\n\n', r.Iall(end));
end

figure('Name', sprintf('pv_check System %d', system), 'Color', 'w');
plot(h.V, h.I, 'k--', 'LineWidth', 1); hold on
plot(r.V, r.I, 'r-', 'LineWidth', 1.8); plot(r.Vmp, r.Imp, 'ro', 'MarkerFaceColor', 'r');
grid on; xlabel('Voltage (V)'); ylabel('Current (A)');
xlim([0, max([h.Voc, r.Voc, 1]) * 1.05]); ylim([0, max([h.Isc, r.Isc, 0.01]) * 1.15]);
legend('healthy (same weather)', 'with the fault', 'maximum power point', 'Location', 'southwest');
title(sprintf('Simscape I-V curve - diagnosis: %s (%.0f %%)', names{r.labelIdx}, 100 * max(r.probs)));

figure('Name', sprintf('pv_check System %d - SHAP and protection', system), 'Color', 'w');
subplot(1, 2, 1);
[~, ord] = sort(abs(r.shap), 'descend'); ord = flip(ord(1:10));
barh(r.shap(ord)); set(gca, 'YTick', 1:10, 'YTickLabel', info.feature_order(ord), 'TickLabelInterpreter', 'none');
grid on; xlabel('SHAP value (+ supports, - against)'); title(sprintf('Why "%s"? (top 10 inputs)', names{r.labelIdx}));
subplot(1, 2, 2);
plot(r.t, r.Iall, 'b-', 'LineWidth', 1.5); hold on
if ~isempty(r.contactor)
    yyaxis right; stairs(r.t(1:numel(r.contactor)), r.contactor, 'r-', 'LineWidth', 1.5); ylim([-0.1 1.2]); ylabel('contactor K1 (1 closed, 0 open)');
    yyaxis left;
end
grid on; xlabel('time (s)'); ylabel('array current (A)');
xline(1, 'k:', 'sweep done -> diagnosis'); title('Array current: I-V sweep, then working point');

last = struct('system', system, 'config', o.config, 'G', o.G, 'Ta', o.Ta, 'Tc', o.Tc, 'V', r.V, 'I', r.I, ...
    'Voc', r.Voc, 'Isc', r.Isc, 'Vmp', r.Vmp, 'Imp', r.Imp, 'inputs', r.x, 'label', names{r.labelIdx}, ...
    'probs', r.probs, 'model', o.model, 'shap', r.shap, 'trip', r.trip);
assignin('base', 'pvlast', last);
end
