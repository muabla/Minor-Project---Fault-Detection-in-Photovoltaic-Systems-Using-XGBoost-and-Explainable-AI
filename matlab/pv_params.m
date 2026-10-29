function [pvp, lay] = pv_params(system, config, scen, ex)
%PV_PARAMS  Values for every block of pv_systemN.slx for one scenario (the circuit itself never changes).
%   scen fields: G, Ta, Tc, light (modules x cells, fraction of G; [] = all 1), short ([branch module substring],
%                substring 0 = whole module), open (branch, 0 = none), corrosion (fraction), rbranch
%   ex : expected-healthy values [Voc_n Isc_n Vmp_n Imp_n FF Rs_n] for the Pi's ratio inputs (zeros if unknown)
%   lay: the layout (module list, slots) that build_pv_slx uses -> same indexing in both.
info = pv_get('pv_info');
C = info.systems.(sprintf('s%d', system)).configs.(config);
mods = C.modules; nM = numel(mods); K = info.constants.slots_per_substring;
nb = max(double(mods(1).n_bypass), 1);           % substrings per module (System 1: whole module = 1)
nSlot = nM * nb * K; nBr = double(C.strings);

lay.mods = mods; lay.nM = nM; lay.nb = nb; lay.K = K; lay.nBr = nBr; lay.bypass = mods(1).n_bypass > 0;
pvp = struct();
pvp.G = scen.G; pvp.Ta = scen.Ta; pvp.Tc = scen.Tc;
pvp.maxVoc = C.max_voc; pvp.maxIsc = C.max_isc; pvp.nStr = nBr; pvp.ex = reshape(ex, 1, 6);
pvp.ns = zeros(1, nSlot); pvp.Ir = zeros(1, nSlot);
pvp.Is = zeros(1, nSlot); pvp.Iph = zeros(1, nSlot); pvp.N = zeros(1, nSlot); pvp.TX = zeros(1, nSlot);
pvp.Rs = zeros(1, nSlot); pvp.Rp = zeros(1, nSlot); pvp.Vbd = zeros(1, nSlot);
pvp.Rshort = 1e9 * ones(1, nM * nb);             % fault resistors across each substring: 1e9 ohm = healthy
pvp.Ropen = 1e-6 * ones(1, nBr);                 % series fault resistor per string: 1e9 ohm = wire cut
pvp.Rcorr = 1e-6 * ones(1, nBr);                 % series resistance per string: corroded connectors
light = scen.light;
if isempty(light), light = ones(nM, double(mods(1).n_cells)); end
assert(size(light, 1) == nM, 'light must have one row per module (%d)', nM);

k = 0;
for j = 1:nM
    m = mods(j); nc = double(m.n_cells);
    groups = split_cells(nc, nb);
    for g = 1:nb
        [cnt, val] = slot_bins(light(j, groups{g}), K);
        for s = 1:K
            k = k + 1;
            pvp.ns(k) = cnt(s); pvp.Ir(k) = scen.G * val(s);
            pvp.Is(k) = m.I0_ref; pvp.Iph(k) = m.Isc; pvp.N(k) = m.n_ideal; pvp.TX(k) = 3 * m.n_ideal;
            pvp.Rs(k) = m.Rs_cell; pvp.Rp(k) = m.Rsh_cell;
            pvp.Vbd(k) = info.constants.V_breakdown * cnt(s);
        end
    end
end

% faults
br = double([mods.branch]);
for r = 1:size(scen.short, 1)
    b = scen.short(r, 1); mi = scen.short(r, 2); s = scen.short(r, 3);
    idx = find(br == b); j = idx(mi);                    % flat module index
    if s == 0
        pvp.Rshort((j - 1) * nb + (1:nb)) = 1e-3;        % whole module shorted
    else
        pvp.Rshort((j - 1) * nb + s) = 1e-3;             % failed-short bypass diode -> substring shorted
    end
end
if scen.open > 0, pvp.Ropen(scen.open) = 1e9; end
if scen.corrosion > 0
    idx = find(br == scen.rbranch);
    pvp.Rcorr(scen.rbranch) = scen.corrosion * sum([mods(idx).Vmp]) / min([mods(idx).Imp]);
end
% I-V sweep: 0 -> 1.15 x (sum of module Voc) in 1 s, sampled every 1 ms (1001 points).
% After the sweep (1.0 - 1.6 s) the tracer holds the array near its working voltage, like an inverter load:
% current keeps flowing, so the contactor trip (short circuit) is visible as the current dropping to zero.
pvp.Tsweep = 1; pvp.Vsweep = 1.15 * sum([mods(br == 1).Voc]);
pvp.Vop = 0.76 * sum([mods(br == 1).Voc]);
pvp.tb = [0 1 1.05 2]; pvp.vb = [0 pvp.Vsweep pvp.Vop pvp.Vop];
pvp.doShap = 0;                                  % SHAP block inside the Pi subsystem: 1 = compute
end

function groups = split_cells(nc, nb)
% same split as numpy.array_split(range(nc), nb)
groups = cell(1, nb); base = floor(nc / nb); extra = mod(nc, nb); i0 = 0;
for g = 1:nb
    n = base + (g <= extra);
    groups{g} = i0 + (1:n); i0 = i0 + n;
end
end
