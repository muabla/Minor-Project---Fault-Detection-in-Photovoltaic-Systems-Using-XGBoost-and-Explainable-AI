function lines = pv_reasons(x, xHealthy, phi, n)
%PV_REASONS  The n inputs with the largest SHAP values, as plain-language sentences (same wording as the website).
%   x, xHealthy : 22 inputs of this reading and of the healthy array in the same weather;  phi : SHAP values
if nargin < 4, n = 6; end
words = {'open-circuit voltage', 'short-circuit current', 'irradiance', 'ambient temperature', 'voltage at max power', ...
    'current at max power', 'fill factor', 'current per unit sunlight', 'knee position (Vmp/Voc)', 'MPP current vs Isc', ...
    'MPP current per unit sunlight', 'curve slope near Voc', 'curve slope near Isc', 'steps in the I-V curve', ...
    'voltage (Voc)', 'current (Isc)', 'voltage at max power', 'current at max power', 'fill factor', 'power', ...
    'series resistance', 'number of parallel strings'};
phi = phi(:)'; x = x(:)'; xHealthy = xHealthy(:)';
[~, order] = sort(abs(phi), 'descend'); order = order(:)';
lines = strings(0, 1);
for q = order(1:min(n, numel(order)))
    v = x(q); dir = 'supports'; if phi(q) < 0, dir = 'argues against'; end
    W = [upper(words{q}(1)) words{q}(2:end)];
    if q == 14
        s = sprintf('The I-V curve has %d power peak(s); a healthy array has %d.', round(v), round(xHealthy(q)));
        if v > xHealthy(q), s = [s ' Extra peaks mean bypass diodes have switched on.']; end
    elseif q == 22
        s = sprintf('The array has %d parallel string(s) (set at installation).', round(v));
    elseif q == 3 || q == 4
        s = sprintf('Weather input (%s) - only the reference for what is expected.', words{q});
    else
        if q >= 15, ref = 1; else, ref = xHealthy(q); end      % *_r inputs are already "measured / healthy"
        if q == 21 || q == 12
            r = v / max(ref, 1e-9);
            if q == 12, nm = 'The curve slope near Voc (series resistance)'; else, nm = 'Series resistance'; end
            if r > 1.15, s = sprintf('%s is %.1fx what a healthy array has.', nm, r);
            else, s = sprintf('%s is normal (%.2fx healthy).', nm, r); end
        elseif abs(ref) < 1e-9
            s = sprintf('%s: %.3f.', W, v);
        else
            pc = (v / ref - 1) * 100;
            if abs(pc) < 1.5
                s = sprintf('%s is normal (within %.1f %% of a healthy array).', W, abs(pc));
            elseif pc < 0
                s = sprintf('%s is %.0f %% below what a healthy array gives in this sunlight and temperature.', W, -pc);
            else
                s = sprintf('%s is %.0f %% above what a healthy array gives in this sunlight and temperature.', W, pc);
            end
        end
    end
    lines(end + 1, 1) = sprintf('%-+8.3f [%s]  %s', phi(q), dir, s); %#ok<AGROW>
end
end
