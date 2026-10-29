function mdl = build_pv_slx(system, config)
%BUILD_PV_SLX  Creates the Simulink/Simscape model of one PV system:
%      [PV array]  subsystem: Module 1..n boxes (Solar Cell blocks, bypass diodes, FAULT resistors)
%   -> [DC contactor K1]  opened by the Raspberry Pi on a short circuit
%   -> [I-V tracer] (voltage sweep, then holds the working voltage) + voltage/current sensors
%   -> [Raspberry Pi 4 - fault monitor]  22 inputs -> XGBoost Predict -> SHAP explanation, protection (trip)
%   -> displays: class, probabilities, contactor state; scope of the array current
%   build_pv_slx(3)   build_pv_slx(2)   build_pv_slx(1,'series')   build_pv_slx(1,'parallel')
%   Saves pv_system3.slx etc. in this folder.
if nargin < 2, config = 'series'; end
here = fileparts(mfilename('fullpath'));
P = pv_ports();
if ~isfield(P, 'sw') || isempty(P.sw), P = pv_ports(true); end      % contactor port test added later
L = P.lib;
if isempty(P.cellprm), P.cellprm = 'ee.enum.sources.solar_cell_prm.eightParameter'; end   % full model with Rp
info = pv_get('pv_info');
mdl = pv_model_name(system, config);
fprintf('\n=== build_pv_slx: %s ===\n', mdl);
if bdIsLoaded(mdl), close_system(mdl, 0); end
f = fullfile(here, [mdl '.slx']);
if exist(f, 'file'), delete(f); end
new_system(mdl);
set_param(mdl, 'Solver', 'ode23t', 'StopTime', '1.6005', 'MaxStep', '1e-3', 'RelTol', '1e-4', ...
    'ReturnWorkspaceOutputs', 'on');

heat = info.systems.(sprintf('s%d', system)).heat;
scen = struct('G', 800, 'Ta', 25, 'Tc', 25 + heat * 800, 'light', [], 'short', zeros(0, 3), 'open', 0, ...
    'corrosion', 0, 'rbranch', 1);
[pvp, lay] = pv_params(system, config, scen, zeros(1, 6));
assignin('base', 'pvp', pvp);
sysname = info.systems.(sprintf('s%d', system)).name;

% ================================================================ PV array subsystem
arr = [mdl '/PV array'];
hArr = new_sub(arr, [80, 120, 260, 300]);
[~, pPlus] = conn_port(arr, '+', 'Right', 1, [900, 40]);
[~, pMinus] = conn_port(arr, '-', 'Right', 2, [900, 300]);
br = double([lay.mods.branch]);
for b = 1:lay.nBr
    idx = find(br == b); prevN = [];
    yb = 60 + (b - 1) * 160;
    for q = 1:numel(idx)
        j = idx(q);
        ms = [arr sprintf('/Module %d', j)];
        hM = new_sub(ms, [60 + (q - 1) * 120, yb, 140 + (q - 1) * 120, yb + 70]);
        build_module(ms, j, lay, P, L);
        [mP_, mN_] = sub_ports(hM);
        if q == 1, firstP = mP_; else, ln(arr, prevN, mP_); end
        prevN = mN_;
    end
    xs = 80 + numel(idx) * 120;
    ro = ab(arr, L.res, sprintf('FAULT open wire string %d', b), [xs, yb - 40, 50, 20]);
    set_param(ro, 'R', sprintf('pvp.Ropen(%d)', b));
    rc = ab(arr, L.res, sprintf('FAULT corrosion string %d', b), [xs + 80, yb - 40, 50, 20]);
    set_param(rc, 'R', sprintf('pvp.Rcorr(%d)', b));
    ln(arr, firstP, pt(ro, {'LConn', 1})); ln(arr, pt(ro, {'RConn', 1}), pt(rc, {'LConn', 1}));
    endP = pt(rc, {'RConn', 1});
    if lay.nBr > 1                                 % blocking diode per string
        d = ab(arr, L.diode, sprintf('Blocking diode string %d', b), [xs + 160, yb - 45, 30, 30]);
        set_param(d, 'Vf', '1e-3', 'Ron', '1e-3', 'Goff', '1e-9');
        if P.diode_fwd_at_p
            ln(arr, endP, pt(d, {'LConn', 1})); endP = pt(d, {'RConn', 1});
        else
            ln(arr, endP, pt(d, {'RConn', 1})); endP = pt(d, {'LConn', 1});
        end
    end
    ln(arr, endP, pPlus); ln(arr, prevN, pMinus);
end
note(arr, sprintf(['%s\nEach Module box: 3 substrings x 4 Solar Cell blocks ("slots", cells sorted by light), a bypass diode ' ...
    'per substring and a FAULT resistor per substring (short).\nString faults: open wire, corrosion (series resistance).'], sysname), [60, 0]);

% ================================================================ contactor, tracer, sensors
x1 = 340; y1 = 120;
[aPlus, aMinus] = sub_ports(hArr);
sw = ab(mdl, L.sw, 'DC contactor K1', [x1, y1 + 10, 50, 40]);
cs = ab(mdl, L.cs, 'Current sensor', [x1 + 110, y1, 40, 40]);
cvs = ab(mdl, L.cvs, 'I-V tracer (voltage sweep)', [x1 + 230, y1 + 60, 50, 60]);
vs = ab(mdl, L.vs, 'Voltage sensor', [x1 + 120, y1 + 90, 40, 50]);
ref = ab(mdl, L.ref, 'Ground', [x1 + 240, y1 + 200, 30, 30]);
sol = ab(mdl, L.solver, 'Solver Configuration', [x1 + 320, y1 + 200, 60, 30]);
if P.cvs_a_to_p, cp = P.cvs.a; cn = P.cvs.b; else, cp = P.cvs.b; cn = P.cvs.a; end
Tn = pt(ref, {'LConn', 1});
ln(mdl, aPlus, pt(sw, P.sw.a)); ln(mdl, pt(sw, P.sw.b), pt(cs, P.cs.b));
ln(mdl, pt(cvs, cp), pt(cs, P.cs.a)); ln(mdl, pt(cvs, cn), Tn); ln(mdl, aMinus, Tn);
ln(mdl, pt(sol, {'RConn', 1}), Tn);
ln(mdl, pt(vs, P.vs.a), pt(cs, P.cs.b)); ln(mdl, pt(vs, P.vs.b), Tn);

% tracer command: ramp 0 -> Vsweep in 1 s, then hold Vop (lookup table over time)
clk = ab(mdl, 'simulink/Sources/Clock', 'Clock', [x1 + 60, y1 - 120, 30, 30]);
lut = ab(mdl, 'simulink/Lookup Tables/1-D Lookup Table', 'Tracer voltage command', [x1 + 120, y1 - 125, 70, 40]);
set_param(lut, 'Table', 'pvp.vb', 'BreakpointsForDimension1', 'pvp.tb');
try, set_param(lut, 'ExtrapMethod', 'Clip'); catch, end
s2p = ab(mdl, L.sl2ps, 'S-PS sweep', [x1 + 230, y1 - 120, 30, 30]);
try, set_param(s2p, 'Unit', 'V'); catch, end
ln(mdl, pt(clk, {'Outport', 1}), pt(lut, {'Inport', 1})); ln(mdl, pt(lut, {'Outport', 1}), pt(s2p, {'Inport', 1}));
ln(mdl, pt(s2p, {'RConn', 1}), pt(cvs, P.cvs.in));

% sensors -> Simulink signals sampled every 1 ms (the tracer's ADC)
ci = ab(mdl, L.ps2sl, 'PS-S current', [x1 + 200, y1 - 20, 30, 20]); try, set_param(ci, 'Unit', 'A'); catch, end
cvv = ab(mdl, L.ps2sl, 'PS-S voltage', [x1 + 200, y1 + 160, 30, 20]); try, set_param(cvv, 'Unit', 'V'); catch, end
ln(mdl, pt(cs, P.cs.out), pt(ci, {'LConn', 1})); ln(mdl, pt(vs, P.vs.out), pt(cvv, {'LConn', 1}));
zi = ab(mdl, 'simulink/Discrete/Zero-Order Hold', 'ADC current (1 ms)', [x1 + 380, y1 - 25, 40, 30]);
zv = ab(mdl, 'simulink/Discrete/Zero-Order Hold', 'ADC voltage (1 ms)', [x1 + 380, y1 + 155, 40, 30]);
set_param(zi, 'SampleTime', '1e-3'); set_param(zv, 'SampleTime', '1e-3');
ln(mdl, pt(ci, {'Outport', 1}), pt(zi, {'Inport', 1})); ln(mdl, pt(cvv, {'Outport', 1}), pt(zv, {'Inport', 1}));
wi = tows(mdl, 'pv_I', [x1 + 470, y1 - 90, 50, 25]); wv = tows(mdl, 'pv_V', [x1 + 470, y1 + 220, 50, 25]);
ln(mdl, pt(zi, {'Outport', 1}), pt(wi, {'Inport', 1})); ln(mdl, pt(zv, {'Outport', 1}), pt(wv, {'Inport', 1}));
sc = ab(mdl, 'simulink/Sinks/Scope', 'Array current (A)', [x1 + 470, y1 - 150, 40, 40]);
ln(mdl, pt(zi, {'Outport', 1}), pt(sc, {'Inport', 1}));

% ================================================================ Raspberry Pi 4 - fault monitor
pis = [mdl '/Raspberry Pi 4 - fault monitor'];
hPi = new_sub(pis, [x1 + 560, y1 - 20, x1 + 760, y1 + 180]);
iV = add_block('simulink/Sources/In1', [pis '/V'], 'Position', [30 40 60 55]);
iI = add_block('simulink/Sources/In1', [pis '/I'], 'Position', [30 90 60 105]);
fb = [pis '/Read I-V sweep and compute 22 inputs'];
add_block('simulink/User-Defined Functions/MATLAB Function', fb, 'Position', [200 30 380 280]);
set_code(fb, pi_code());
hF = get_param(fb, 'Handle');
ln(pis, pt(iV, {'Outport', 1}), pt(hF, {'Inport', 1})); ln(pis, pt(iI, {'Outport', 1}), pt(hF, {'Inport', 2}));
consts = {'G (pyranometer)', 'pvp.G'; 'Ta (temperature sensor)', 'pvp.Ta'; 'MaxVoc', 'pvp.maxVoc'; ...
          'MaxIsc', 'pvp.maxIsc'; 'Parallel strings', 'pvp.nStr'; 'Expected healthy', 'pvp.ex'};
for q = 1:size(consts, 1)
    hc = add_block('simulink/Sources/Constant', [pis '/' consts{q, 1}], 'Value', consts{q, 2}, ...
        'Position', [60, 120 + 35 * q, 130, 140 + 35 * q]);
    ln(pis, pt(hc, {'Outport', 1}), pt(hF, {'Inport', 2 + q}));
end
if ~isempty(P.xgb)
    hx = add_block(P.xgb, [pis '/XGBoost Predict'], 'Position', [470 40 600 140]);
    set_xgb_block(hx);
else
    add_block('simulink/User-Defined Functions/MATLAB Function', [pis '/XGBoost Predict'], 'Position', [470 40 600 140]);
    set_code([pis '/XGBoost Predict'], xgb_fallback_code());
    hx = get_param([pis '/XGBoost Predict'], 'Handle');
end
ln(pis, pt(hF, {'Outport', 1}), pt(hx, {'Inport', 1}));
% SHAP explanation (MATLAB shapley on the same model, computed once when the sweep is complete)
sb = [pis '/SHAP explanation'];
add_block('simulink/User-Defined Functions/MATLAB Function', sb, 'Position', [680 170 820 260]);
set_code(sb, shap_code());
hS = get_param(sb, 'Handle');
hEn = add_block('simulink/Sources/Constant', [pis '/SHAP on'], 'Value', 'pvp.doShap', 'Position', [560 260 610 280]);
ln(pis, pt(hF, {'Outport', 1}), pt(hS, {'Inport', 1})); ln(pis, pt(hx, {'Outport', 1}), pt(hS, {'Inport', 2}));
ln(pis, pt(hF, {'Outport', 2}), pt(hS, {'Inport', 3})); ln(pis, pt(hEn, {'Outport', 1}), pt(hS, {'Inport', 4}));
% protection: trip the DC contactor on a short circuit (latched until the next run)
pb = [pis '/Protection logic (trip on short circuit)'];
add_block('simulink/User-Defined Functions/MATLAB Function', pb, 'Position', [680 300 820 360]);
set_code(pb, protection_code());
hP = get_param(pb, 'Handle');
ln(pis, pt(hx, {'Outport', 1}), pt(hP, {'Inport', 1})); ln(pis, pt(hF, {'Outport', 2}), pt(hP, {'Inport', 2}));
outs = {'class', hx, 1; 'probabilities', hx, 2; '22 inputs', hF, 1; 'SHAP', hS, 1; 'trip', hP, 1};
for q = 1:size(outs, 1)
    ho = add_block('simulink/Sinks/Out1', [pis '/' outs{q, 1}], 'Position', [900, 40 + 70 * (q - 1), 930, 55 + 70 * (q - 1)]);
    ln(pis, pt(outs{q, 2}, {'Outport', outs{q, 3}}), pt(ho, {'Inport', 1}));
end

phP = get_param(hPi, 'PortHandles');
ln(mdl, pt(zv, {'Outport', 1}), phP.Inport(1)); ln(mdl, pt(zi, {'Outport', 1}), phP.Inport(2));
xd = x1 + 820;
dL = ab(mdl, 'simulink/Sinks/Display', 'Predicted class (0 normal 1 shading 2 dirt 3 open 4 short 5 degradation)', [xd, y1 - 30, 80, 30]);
dS = ab(mdl, 'simulink/Sinks/Display', 'Class probabilities', [xd, y1 + 20, 220, 30]);
ln(mdl, phP.Outport(1), pt(dL, {'Inport', 1})); ln(mdl, phP.Outport(2), pt(dS, {'Inport', 1}));
names = {'pv_label', 1; 'pv_score', 2; 'pv_x', 3; 'pv_shap', 4; 'pv_trip', 5};
for q = 1:size(names, 1)
    w = tows(mdl, names{q, 1}, [xd + 260, y1 - 30 + 45 * (q - 1), 60, 25]);
    ln(mdl, phP.Outport(names{q, 2}), pt(w, {'Inport', 1}));
end
% trip -> relay delay 0.1 s -> contactor command (1 = closed, 0 = open)
dly = ab(mdl, 'simulink/Discrete/Delay', 'Relay delay 0.1 s', [xd, y1 + 180, 50, 30]);
set_param(dly, 'DelayLength', '100');
ng = ab(mdl, 'simulink/Math Operations/Gain', 'invert', [xd + 80, y1 + 180, 30, 30]); set_param(ng, 'Gain', '-1');
bs = ab(mdl, 'simulink/Math Operations/Bias', 'command 1=closed', [xd + 140, y1 + 180, 40, 30]); set_param(bs, 'Bias', '1');
dC = ab(mdl, 'simulink/Sinks/Display', 'Contactor K1 (1 = closed, 0 = OPEN)', [xd + 220, y1 + 240, 60, 30]);
wc = tows(mdl, 'pv_contactor', [xd + 220, y1 + 290, 60, 25]);
s2c = ab(mdl, L.sl2ps, 'S-PS contactor', [x1 + 10, y1 + 260, 30, 30]);
ln(mdl, phP.Outport(5), pt(dly, {'Inport', 1})); ln(mdl, pt(dly, {'Outport', 1}), pt(ng, {'Inport', 1}));
ln(mdl, pt(ng, {'Outport', 1}), pt(bs, {'Inport', 1})); ln(mdl, pt(bs, {'Outport', 1}), pt(dC, {'Inport', 1}));
ln(mdl, pt(bs, {'Outport', 1}), pt(wc, {'Inport', 1})); ln(mdl, pt(bs, {'Outport', 1}), pt(s2c, {'Inport', 1}));
ln(mdl, pt(s2c, {'RConn', 1}), pt(sw, P.sw.ctrl));

note(mdl, sprintf(['%s\nPV array -> DC contactor K1 -> I-V tracer (0-1 s: voltage sweep; 1-1.6 s: holds the working voltage).\n' ...
    'The Raspberry Pi subsystem computes the 22 inputs, runs the XGBoost model trained in Python, explains it with SHAP, ' ...
    'and opens K1 on a short circuit.\nRun it with pv_check(...) (values come from the workspace variable pvp).'], sysname), [80, 20]);

fprintf('Compiling the model (Update Diagram)...\n');
set_param(mdl, 'SimulationCommand', 'update');
save_system(mdl, f);
fprintf('Saved %s  (%d modules, %d Solar Cell blocks, %d strings)\n=== build OK ===\n\n', f, lay.nM, lay.nM * lay.nb * lay.K, lay.nBr);
open_system(mdl);
end

% ================================================================ one module (inside its subsystem)
function build_module(ms, j, lay, P, L)
[~, mP] = conn_port(ms, 'P', 'Left', 1, [10, 20]);
[~, mN] = conn_port(ms, 'N', 'Right', 2, [130 + lay.nb * 170, 20]);
W = 60; H = 60; k = (j - 1) * lay.nb * lay.K; prevN = mP; firstP = [];
for g = 1:lay.nb
    xs = 80 + (g - 1) * 170; ys = 60;
    lastN = [];
    for s = 1:lay.K
        k = k + 1; yy = ys + (s - 1) * (H + 30);
        c = ab(ms, L.cell, sprintf('S%d cell slot %d', g, s), [xs, yy, W * 0.7, H * 0.8]);
        set_param(c, 'prm', P.cellprm);
        setp(c, 'N_series', sprintf('pvp.ns(%d)', k), 'Is', sprintf('pvp.Is(%d)', k), 'Is2', '0', ...
            'Iph', sprintf('pvp.Iph(%d)', k), 'Ir0', '1000', 'ec', sprintf('pvp.N(%d)', k), ...
            'Rs', sprintf('pvp.Rs(%d)', k), 'Rp', sprintf('pvp.Rp(%d)', k), 'TIPH1', '0.0005', 'EG', '1.12', ...
            'TXIS1', sprintf('pvp.TX(%d)', k), 'TRS1', '0', 'TRP1', '0', 'Tmeas', '25', 'TFIXED', 'pvp.Tc');
        ir = ab(ms, L.psc, sprintf('S%d light %d', g, s), [xs - 60, yy, 40, 20]);
        set_param(ir, 'constant', sprintf('pvp.Ir(%d)', k));
        try, set_param(ir, 'constant_unit', 'W/m^2'); catch, end
        ln(ms, pt(ir, {'RConn', 1}), pt(c, P.cell.ir));
        if ~lay.bypass                         % no bypass diodes (System 1): cell reverse breakdown
            d = ab(ms, L.diode, sprintf('S%d breakdown %d', g, s), [xs + W, yy, 30, 30]);
            set_param(d, 'Vf', sprintf('pvp.Vbd(%d)', k), 'Ron', '1e-3', 'Goff', '1e-9');
            reverse_diode(ms, d, pt(c, P.cell.p), pt(c, P.cell.n), P);
        end
        if s == 1, subP = pt(c, P.cell.p); else, ln(ms, lastN, pt(c, P.cell.p)); end
        lastN = pt(c, P.cell.n);
    end
    subN = lastN;
    if lay.bypass
        d = ab(ms, L.diode, sprintf('S%d bypass diode', g), [xs + W + 10, ys + lay.K * (H + 30) / 2 - 15, 30, 30]);
        set_param(d, 'Vf', '0.5', 'Ron', '1e-3', 'Goff', '1e-9');
        reverse_diode(ms, d, subP, subN, P);
    end
    ln(ms, prevN, subP);
    if g == 1, firstP = subP; end
    prevN = subN;
    if lay.bypass                              % FAULT: failed-short bypass diode (1e9 ohm healthy, 1 mohm shorted)
        r = ab(ms, L.res, sprintf('FAULT short S%d', g), [xs, ys + lay.K * (H + 30), 50, 20]);
        set_param(r, 'R', sprintf('pvp.Rshort(%d)', (j - 1) * lay.nb + g));
        ln(ms, pt(r, {'LConn', 1}), subP); ln(ms, pt(r, {'RConn', 1}), subN);
    end
end
if ~lay.bypass                                 % FAULT: whole module shorted (System 1)
    r = ab(ms, L.res, 'FAULT short module', [80, 60 + lay.K * (H + 30), 50, 20]);
    set_param(r, 'R', sprintf('pvp.Rshort(%d)', j));
    ln(ms, pt(r, {'LConn', 1}), firstP); ln(ms, pt(r, {'RConn', 1}), prevN);
end
ln(ms, prevN, mN);
end

% ================================================================ code of the MATLAB Function blocks
function s = pi_code()
s = sprintf([ ...
'function [x, done] = pi_monitor(V, I, G, Ta, maxVoc, maxIsc, nStr, ex)\n' ...
'%%#codegen\n' ...
'%% Raspberry Pi program: collects the I-V tracer samples (one sweep = 1001 samples, 0-1 s),\n' ...
'%% then computes the 22 model inputs (pv_features.m, same calculation as the Python simulator).\n' ...
'N = 1001;\n' ...
'persistent Vb Ib k xh\n' ...
'if isempty(k)\n' ...
'    Vb = zeros(N, 1); Ib = zeros(N, 1); k = 0; xh = zeros(1, 22);\n' ...
'end\n' ...
'if k < N\n' ...
'    k = k + 1; Vb(k) = V; Ib(k) = I;\n' ...
'    if k == N\n' ...
'        sv = 1; if Vb(N) < 0, sv = -1; end\n' ...
'        si = 1; if Ib(1) < 0, si = -1; end\n' ...
'        xh = pv_features(sv * Vb, si * Ib, G, Ta, maxVoc, maxIsc, nStr, ex);\n' ...
'    end\n' ...
'end\n' ...
'x = xh;\n' ...
'done = double(k >= N);\n' ...
'end\n']);
end

function s = shap_code()
s = sprintf([ ...
'function phi = shap_explanation(x, label, done, enable)\n' ...
'%% SHAP: contribution of each of the 22 inputs to the predicted class (MATLAB shapley, same XGBoost model).\n' ...
'%% Computed once, when the sweep is complete. Simulation only (calls MATLAB; not part of the Pi code).\n' ...
'coder.extrinsic(''pv_shap_extrinsic'');\n' ...
'persistent ph got\n' ...
'if isempty(got), ph = zeros(1, 22); got = false; end\n' ...
'if done > 0.5 && ~got && enable > 0.5\n' ...
'    tmp = zeros(1, 22);\n' ...
'    tmp = pv_shap_extrinsic(x, label);\n' ...
'    ph = tmp; got = true;\n' ...
'end\n' ...
'phi = ph;\n' ...
'end\n']);
end

function s = protection_code()
s = sprintf([ ...
'function trip = protection(label, done)\n' ...
'%%#codegen\n' ...
'%% Short circuit (class 4) -> trip the DC contactor. Latched: stays open until the next run (RESET).\n' ...
'persistent t\n' ...
'if isempty(t), t = 0; end\n' ...
'if done > 0.5 && double(label) == 4, t = 1; end\n' ...
'trip = t;\n' ...
'end\n']);
end

function s = xgb_fallback_code()
s = sprintf([ ...
'function [label, score] = xgb_predict(x)\n' ...
'%% Calls predict() of the imported XGBoost model (workspace variable xgbMdl).\n' ...
'coder.extrinsic(''pv_predict_extrinsic'');\n' ...
'label = 0; score = zeros(1, 6);\n' ...
'[label, score] = pv_predict_extrinsic(x);\n' ...
'end\n']);
end

function set_code(blockPath, code)
ch = find(sfroot, '-isa', 'Stateflow.EMChart', 'Path', blockPath);
ch.Script = code;
end

function set_xgb_block(h)
% the block reads the model from the workspace variable named in its dialog (default xgbMdl, created by setup_pv);
% switch on the extra output port with the class probabilities
dp = get_param(h, 'DialogParameters'); f = fieldnames(dp);
for q = 1:numel(f)
    pr = lower(dp.(f{q}).Prompt);
    try, v = get_param(h, f{q}); catch, v = ''; end
    if ischar(v) && contains(pr, 'score') && any(strcmpi(v, {'on', 'off'}))
        set_param(h, f{q}, 'on');
    end
end
end

% ================================================================ helpers
function h = new_sub(path, pos)
h = add_block('simulink/Ports & Subsystems/Subsystem', path, 'Position', pos);
try, delete_line(path, 'In1/1', 'Out1/1'); catch, end
try, delete_block([path '/In1']); catch, end
try, delete_block([path '/Out1']); catch, end
end

function [h, p] = conn_port(sub, name, side, num, xy)
h = add_block('nesl_utility/Connection Port', [sub '/' name], 'Position', [xy(1), xy(2), xy(1) + 30, xy(2) + 16]);
set_param(h, 'Port', num2str(num));
done = false;
for v = {side, lower(side), upper(side)}
    try, set_param(h, 'Side', v{1}); done = true; break, catch, end
end
if ~done, fprintf('  (note: could not place port %s on the %s side; it stays on the left)\n', name, side); end
ph = get_param(h, 'PortHandles');
if ~isempty(ph.RConn), p = ph.RConn(1); else, p = ph.LConn(1); end
end

function [pA, pB] = sub_ports(h)
% the subsystem's two Simscape terminals in port-number order (works whichever side they ended up on)
ph = get_param(h, 'PortHandles');
c = [ph.LConn(:); ph.RConn(:)];
if numel(c) < 2, error('build_pv_slx: subsystem %s has %d terminals (expected 2)', getfullname(h), numel(c)); end
if numel(ph.LConn) == 1 && numel(ph.RConn) == 1
    pA = ph.LConn(1); pB = ph.RConn(1);              % module: P on the left, N on the right
elseif numel(ph.RConn) == 2
    pA = ph.RConn(1); pB = ph.RConn(2);              % array: + and - on the right
else
    pA = ph.LConn(1); pB = ph.LConn(2);              % both ended up on the left
end
end

function reverse_diode(m, d, nodeP, nodeN, P)
% diode that does NOT conduct the cells' own current (bypass / breakdown): anode on the - side
if P.diode_fwd_at_p
    ln(m, pt(d, {'LConn', 1}), nodeN); ln(m, pt(d, {'RConn', 1}), nodeP);
else
    ln(m, pt(d, {'LConn', 1}), nodeP); ln(m, pt(d, {'RConn', 1}), nodeN);
end
end

function setp(h, varargin)
for q = 1:2:numel(varargin)
    try
        set_param(h, varargin{q}, varargin{q + 1});
    catch e
        error('build_pv_slx: could not set %s on %s: %s', varargin{q}, getfullname(h), e.message);
    end
end
end

function h = ab(m, src, name, xywh)
h = add_block(src, [m '/' name], 'MakeNameUnique', 'on', 'Position', [xywh(1), xywh(2), xywh(1) + xywh(3), xywh(2) + xywh(4)]);
end

function w = tows(m, var, xywh)
w = ab(m, 'simulink/Sinks/To Workspace', var, xywh);
set_param(w, 'VariableName', var, 'SaveFormat', 'Array');
end

function p = pt(h, spec)
ph = get_param(h, 'PortHandles');
p = ph.(spec{1})(spec{2});
end

function ln(m, a, b)
add_line(m, a, b, 'autorouting', 'on');
end

function note(m, txt, xy)
try
    a = Simulink.Annotation([m '/' strrep(txt, '/', '//')]);
    a.Position = [xy(1), xy(2), xy(1) + 900, xy(2) + 60];
catch
end
end
