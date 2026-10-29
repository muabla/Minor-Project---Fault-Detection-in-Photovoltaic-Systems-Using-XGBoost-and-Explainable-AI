function [x, raw] = pv_features(V, I, G, Ta, maxVoc, maxIsc, nStrings, ex)
%PV_FEATURES  The 22 model inputs from one I-V sweep (same calculation as pv_simulator_v3.py).
%   V, I     : sweep samples, V rising from 0 (array voltage, array current)
%   G, Ta    : irradiance (W/m2) and air temperature (C) from the sensors
%   maxVoc, maxIsc, nStrings : installation data (normalisation limits, number of parallel strings)
%   ex       : what a HEALTHY array gives in the same sunlight/temperature:
%              [Voc_n Isc_n Vmp_n Imp_n FF Rs_n]  (all zeros -> ratio inputs are 0)
%   x        : 1x22 inputs in the model's order
%              Voc/MaxVoc Isc/MaxIsc G/1000 AT/50 Vmp/MaxVoc Imp/MaxIsc FF Isc/G Vmp/Voc Imp/Isc Imp/G
%              Rs_n Rsh_n Steps Voc_r Isc_r Vmp_r Imp_r FF_r P_r Rs_r Strings
%   raw      : [Voc Isc Vmp Imp] in volts / amperes
%   Fixed-size arrays only, so it also runs inside a Simulink MATLAB Function block (the Raspberry Pi subsystem).
%#codegen
n = numel(V);
x = zeros(1, 22); raw = zeros(1, 4);
Gn = G / 1000; Tn = Ta / 50;

% ---- Isc (current at V = 0) and Voc (where the current reaches 0)
isc = max(I(1), 0); voc = 0; kend = n;
if isc > 1e-4 * maxIsc
    voc = V(n);
    for k = 2:n
        if I(k) <= 0
            voc = V(k - 1) + (V(k) - V(k - 1)) * I(k - 1) / (I(k - 1) - I(k));
            kend = k - 1;
            break
        end
    end
else
    isc = 0;
end

vmp = 0; imp = 0; ff = 0; rs = 0; rsh = 0; steps = 0;
if isc > 0 && voc > 0
    % generating part of the curve: samples 1..kend plus the end point (Voc, 0); m points used
    m = kend + 1;
    Vc = zeros(n + 1, 1); Ic = zeros(n + 1, 1); Vd = zeros(n + 1, 1); Id = zeros(n + 1, 1);
    for k = 1:kend
        Vc(k) = V(k); Ic(k) = max(I(k), 0);
    end
    Vc(m) = voc; Ic(m) = 0;
    % maximum power point
    pbest = -1; kp = 1; isc0 = 0; voc0 = 0;
    for k = 1:m
        if Vc(k) * Ic(k) > pbest, pbest = Vc(k) * Ic(k); kp = k; end
        if Ic(k) > isc0, isc0 = Ic(k); end
        if Vc(k) > voc0, voc0 = Vc(k); end
    end
    vmp = Vc(kp); imp = Ic(kp);
    ff = vmp * imp / (voc * isc);
    % curve shape (pv_simulator_v3.curve_shape, without sensor noise)
    for k = 1:m
        Id(k) = Ic(m + 1 - k); Vd(k) = Vc(m + 1 - k);            % current ascending -> V(I)
    end
    va = lin_interp(0.03 * isc0, Id, Vd, m); vb = lin_interp(0.20 * isc0, Id, Vd, m);
    rs = (va - vb) / (0.17 * isc0) * isc0 / voc0;
    ia = lin_interp(0, Vc, Ic, m); ib = lin_interp(0.25 * voc0, Vc, Ic, m);
    rsh = (ia - ib) / (0.25 * voc0) * voc0 / isc0;
    pw = zeros(100, 1);
    for k = 1:100
        vk = voc0 * (k - 1) / 99;
        pw(k) = vk * lin_interp(vk, Vc, Ic, m);
    end
    p = zeros(100, 1);                                          % 3-point moving average (numpy 'same')
    for k = 1:100
        s = pw(k);
        if k > 1, s = s + pw(k - 1); end
        if k < 100, s = s + pw(k + 1); end
        p(k) = s / 3;
    end
    pmax = max(p);
    for k = 3:98
        if p(k) >= p(k - 1) && p(k) > p(k + 1) && p(k) > 0.1 * pmax
            steps = steps + 1;
        end
    end
end

voc_n = voc / maxVoc; isc_n = isc / maxIsc; vmp_n = vmp / maxVoc; imp_n = imp / maxIsc;
x(1) = voc_n; x(2) = isc_n; x(3) = Gn; x(4) = Tn; x(5) = vmp_n; x(6) = imp_n; x(7) = ff;
x(8) = q(isc_n, Gn); x(9) = q(vmp_n, voc_n); x(10) = q(imp_n, isc_n); x(11) = q(imp_n, Gn);
x(12) = rs; x(13) = rsh; x(14) = steps;
x(15) = q(voc_n, ex(1)); x(16) = q(isc_n, ex(2)); x(17) = q(vmp_n, ex(3)); x(18) = q(imp_n, ex(4));
x(19) = q(ff, ex(5)); x(20) = q(vmp_n * imp_n, ex(3) * ex(4)); x(21) = q(rs, ex(6));
x(22) = nStrings;
raw = [voc isc vmp imp];
end

function r = q(a, b)
if b > 1e-9
    r = a / b;
else
    r = 0;
end
end

function y = lin_interp(xq, x, y0, n)
% numpy.interp on the first n points: x ascending (repeats allowed), clamps outside the range
if xq <= x(1)
    y = y0(1); return
end
if xq >= x(n)
    y = y0(n); return
end
y = y0(n);
for k = 2:n
    if x(k) >= xq
        if x(k) == x(k - 1)
            y = y0(k);
        else
            y = y0(k - 1) + (y0(k) - y0(k - 1)) * (xq - x(k - 1)) / (x(k) - x(k - 1));
        end
        return
    end
end
end
