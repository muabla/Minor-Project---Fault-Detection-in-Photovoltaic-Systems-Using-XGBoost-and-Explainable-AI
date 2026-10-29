function P = pv_ports(force)
%PV_PORTS  Finds, in YOUR MATLAB version, which port of each Simscape block is which (e.g. which Solar Cell
%   terminal is the irradiance input), by building tiny test circuits and simulating them. Runs once; the
%   result is saved in pv_ports.mat and used by build_pv_slx. Run pv_ports(true) to redo it.

here = fileparts(mfilename('fullpath'));
cache = fullfile(here, 'pv_ports.mat');
if nargin < 1, force = false; end
if ~force && exist(cache, 'file')
    S = load(cache); P = S.P; return
end
fprintf('\n=== pv_ports: testing the Simscape blocks (about 1 minute) ===\n');
warning('off', 'Simulink:Commands:FindSystemVariantsOptionRemoval');
load_system('simulink'); load_system('nesl_utility'); load_system('fl_lib'); load_system('ee_lib');

L.cell   = lp('ee_lib', 'Solar Cell');
L.diode  = lp('fl_lib', 'Diode');
L.res    = lp('fl_lib', 'Resistor');
L.cvs    = lp('fl_lib', 'Controlled Voltage Source');
L.cs     = lp('fl_lib', 'Current Sensor');
L.vs     = lp('fl_lib', 'Voltage Sensor');
L.ref    = lp('fl_lib', 'Electrical Reference');
L.solver = lp('nesl_utility', 'Solver Configuration');
L.ps2sl  = lp('nesl_utility', 'PS-Simulink Converter');
L.sl2ps  = lp('nesl_utility', 'Simulink-PS Converter');
L.psc    = lp('fl_lib', 'PS Constant');
P.lib = L;
C3 = {{'LConn', 1}, {'LConn', 2}, {'RConn', 1}};
C3b = {{'LConn', 1}, {'RConn', 1}, {'RConn', 2}};

% ---- 1. Solar Cell: which port is the irradiance input Ir
P.cell = [];
for i = 1:3
    o = C3(setdiff(1:3, i));
    [ok, msg] = trial(@(m) t_cell(m, L, C3{i}, o));
    fprintf('  Solar Cell, Ir = %s%d: %s\n', C3{i}{1}, C3{i}{2}, okmsg(ok, msg));
    if ok, P.cell = struct('ir', {C3{i}}, 'p', {o{1}}, 'n', {o{2}}); break, end
end
% ---- 2. Controlled Voltage Source: which port is the control input
P.cvs = [];
for i = 1:3
    o = C3b(setdiff(1:3, i));
    [ok, msg] = trial(@(m) t_cvs(m, L, C3b{i}, o));
    fprintf('  Controlled Voltage Source, input = %s%d: %s\n', C3b{i}{1}, C3b{i}{2}, okmsg(ok, msg));
    if ok, P.cvs = struct('in', {C3b{i}}, 'a', {o{1}}, 'b', {o{2}}); break, end
end
% ---- 3. Current Sensor: which port is the measurement output
P.cs = [];
for i = [2 3 1]
    o = C3b(setdiff(1:3, i));
    [ok, msg] = trial(@(m) t_sensor(m, L, P, 'cs', C3b{i}, o));
    fprintf('  Current Sensor, output = %s%d: %s\n', C3b{i}{1}, C3b{i}{2}, okmsg(ok, msg));
    if ok, P.cs = struct('out', {C3b{i}}, 'a', {o{1}}, 'b', {o{2}}); break, end
end
% ---- 4. Voltage Sensor
P.vs = [];
for i = [2 3 1]
    o = C3b(setdiff(1:3, i));
    [ok, msg] = trial(@(m) t_sensor(m, L, P, 'vs', C3b{i}, o));
    fprintf('  Voltage Sensor, output = %s%d: %s\n', C3b{i}{1}, C3b{i}{2}, okmsg(ok, msg));
    if ok, P.vs = struct('out', {C3b{i}}, 'a', {o{1}}, 'b', {o{2}}); break, end
end
if isempty(P.cell) || isempty(P.cvs) || isempty(P.cs) || isempty(P.vs)
    error('pv_ports: a block could not be identified. Check the model parameters.');
end

% ---- 5. Diode direction relative to the solar cell's (guessed) + terminal
[ok, out] = trial(@(m) t_diode(m, L, P));
if ~ok, error('pv_ports: diode test failed: %s', out); end
v = out.get('pvt_v'); v = abs(v(end));
P.diode_fwd_at_p = v < 0.3;      % true: diode LConn1 at cell "p" conducts the cell's current (forward)
fprintf('  Diode test: cell voltage %.3f V -> diode LConn1 at cell p is %s\n', v, ternary(P.diode_fwd_at_p, 'FORWARD', 'reverse'));
% ---- 6. Voltage source direction relative to the cell
[ok, out] = trial(@(m) t_cvsdir(m, L, P));
if ~ok, error('pv_ports: source direction test failed: %s', out); end
i6 = out.get('pvt_i'); i6 = abs(i6(end));
P.cvs_a_to_p = i6 < 4.4;         % true: CVS port "a" connected to cell p gives normal forward operation
fprintf('  Source test: current %.2f A at +0.59 V -> CVS port a goes to %s\n', i6, ternary(P.cvs_a_to_p, 'cell p', 'cell n'));

% ---- 7b. Switch (the DC contactor): which port is the control input
L.sw = lp_in('fl_lib', 'Switch', 'Electrical'); P.lib = L; P.sw = [];
specs = port_list(L.sw);
for i = 1:numel(specs)
    o = specs(setdiff(1:numel(specs), i));
    if numel(o) ~= 2, continue, end
    [ok, msg] = trial(@(m) t_switch(m, L, P, specs{i}, o));
    fprintf('  Switch (contactor), control = %s%d: %s\n', specs{i}{1}, specs{i}{2}, okmsg(ok, msg));
    if ok, P.sw = struct('ctrl', {specs{i}}, 'a', {o{1}}, 'b', {o{2}}); break, end
end
if isempty(P.sw), error('pv_ports: switch control port not identified. Check the model parameters.'); end

% ---- 7. Solar Cell parameterisation option for the full (8-parameter, with Rp) model
P.cellprm = '';
try
    [~, s] = enumeration('ee.enum.sources.solar_cell_prm');
    fprintf('  Solar Cell parameterisation options: %s\n', strjoin(s', ', '));
    k = find(contains(lower(s), '8') | contains(lower(s), 'eight'), 1);
    if isempty(k), k = find(contains(lower(s), 'equivalent'), 1, 'last'); end
    if ~isempty(k), P.cellprm = ['ee.enum.sources.solar_cell_prm.' s{k}]; end
catch e
    fprintf('  could not list Solar Cell options: %s\n', e.message);
end
fprintf('  using: %s\n', P.cellprm);

% ---- 8. ClassificationXGBoost Predict block
P.xgb = find_xgb_block();
fprintf('  ClassificationXGBoost Predict block: %s\n', ternary(isempty(P.xgb), 'NOT FOUND (a MATLAB Function block will call predict instead)', strrep(P.xgb, newline, ' ')));

save(cache, 'P');
fprintf('=== pv_ports done, saved to %s ===\n\n', cache);
end

% ------------------------------------------------------------------ test circuits
function t_cell(m, L, ir, o)
c = ab(m, L.cell, 'cell'); s = ab(m, L.psc, 'G', 'constant', '1000');
try, set_param(s, 'constant_unit', 'W/m^2'); catch, end
r = ab(m, L.res, 'R', 'R', '1'); g = ground(m, L);
ln(m, pt(s, {'RConn', 1}), pt(c, ir));
ln(m, pt(c, o{1}), pt(r, {'LConn', 1})); ln(m, pt(c, o{2}), pt(r, {'RConn', 1})); ln(m, pt(c, o{2}), g);
end

function t_cvs(m, L, in, o)
c = ab(m, L.cvs, 'src'); s = ab(m, L.psc, 'U', 'constant', '1');
try, set_param(s, 'constant_unit', 'V'); catch, end
r = ab(m, L.res, 'R', 'R', '1'); g = ground(m, L);
ln(m, pt(s, {'RConn', 1}), pt(c, in));
ln(m, pt(c, o{1}), pt(r, {'LConn', 1})); ln(m, pt(c, o{2}), pt(r, {'RConn', 1})); ln(m, pt(c, o{2}), g);
end

function t_sensor(m, L, P, kind, outp, o)
c = ab(m, L.cvs, 'src'); s = ab(m, L.psc, 'U', 'constant', '1');
try, set_param(s, 'constant_unit', 'V'); catch, end
r = ab(m, L.res, 'R', 'R', '1'); g = ground(m, L);
k = ab(m, L.(kind), 'sensor'); cv = ab(m, L.ps2sl, 'conv'); w = tows(m, 'pvt_y');
ln(m, pt(s, {'RConn', 1}), pt(c, P.cvs.in));
if strcmp(kind, 'cs')                  % in series
    ln(m, pt(c, P.cvs.a), pt(k, o{1})); ln(m, pt(k, o{2}), pt(r, {'LConn', 1}));
else                                   % across the resistor
    ln(m, pt(c, P.cvs.a), pt(r, {'LConn', 1})); ln(m, pt(k, o{1}), pt(r, {'LConn', 1})); ln(m, pt(k, o{2}), pt(r, {'RConn', 1}));
end
ln(m, pt(r, {'RConn', 1}), pt(c, P.cvs.b)); ln(m, pt(c, P.cvs.b), g);
ln(m, pt(k, outp), pt(cv, {'LConn', 1})); ln(m, pt(cv, {'Outport', 1}), pt(w, {'Inport', 1}));
end

function t_diode(m, L, P)
c = ab(m, L.cell, 'cell'); s = ab(m, L.psc, 'G', 'constant', '1000');
try, set_param(s, 'constant_unit', 'W/m^2'); catch, end
d = ab(m, L.diode, 'D', 'Vf', '0.1', 'Ron', '0.01', 'Goff', '1e-8'); g = ground(m, L);
k = ab(m, L.vs, 'vs'); cv = ab(m, L.ps2sl, 'conv'); w = tows(m, 'pvt_v');
ln(m, pt(s, {'RConn', 1}), pt(c, P.cell.ir));
ln(m, pt(d, {'LConn', 1}), pt(c, P.cell.p)); ln(m, pt(d, {'RConn', 1}), pt(c, P.cell.n)); ln(m, pt(c, P.cell.n), g);
ln(m, pt(k, P.vs.a), pt(c, P.cell.p)); ln(m, pt(k, P.vs.b), pt(c, P.cell.n));
ln(m, pt(k, P.vs.out), pt(cv, {'LConn', 1})); ln(m, pt(cv, {'Outport', 1}), pt(w, {'Inport', 1}));
end

function t_cvsdir(m, L, P)
c = ab(m, L.cell, 'cell'); s = ab(m, L.psc, 'G', 'constant', '1000');
try, set_param(s, 'constant_unit', 'W/m^2'); catch, end
src = ab(m, L.cvs, 'src'); u = ab(m, L.psc, 'U', 'constant', '0.59');
try, set_param(u, 'constant_unit', 'V'); catch, end
k = ab(m, L.cs, 'cs'); cv = ab(m, L.ps2sl, 'conv'); w = tows(m, 'pvt_i'); g = ground(m, L);
ln(m, pt(s, {'RConn', 1}), pt(c, P.cell.ir)); ln(m, pt(u, {'RConn', 1}), pt(src, P.cvs.in));
ln(m, pt(src, P.cvs.a), pt(k, P.cs.a)); ln(m, pt(k, P.cs.b), pt(c, P.cell.p));
ln(m, pt(c, P.cell.n), pt(src, P.cvs.b)); ln(m, pt(src, P.cvs.b), g);
ln(m, pt(k, P.cs.out), pt(cv, {'LConn', 1})); ln(m, pt(cv, {'Outport', 1}), pt(w, {'Inport', 1}));
end

function t_switch(m, L, P, ctrl, o)
c = ab(m, L.cvs, 'src'); s = ab(m, L.psc, 'U', 'constant', '1');
try, set_param(s, 'constant_unit', 'V'); catch, end
k = ab(m, L.psc, 'K', 'constant', '1');
r = ab(m, L.res, 'R', 'R', '1'); g = ground(m, L); w = ab(m, L.sw, 'sw');
ln(m, pt(s, {'RConn', 1}), pt(c, P.cvs.in)); ln(m, pt(k, {'RConn', 1}), pt(w, ctrl));
ln(m, pt(c, P.cvs.a), pt(w, o{1})); ln(m, pt(w, o{2}), pt(r, {'LConn', 1}));
ln(m, pt(r, {'RConn', 1}), pt(c, P.cvs.b)); ln(m, pt(c, P.cvs.b), g);
end

function specs = port_list(src)
m = 'pv_ports_tmp2';
if bdIsLoaded(m), close_system(m, 0); end
new_system(m); h = add_block(src, [m '/b']); ph = get_param(h, 'PortHandles'); specs = {};
for t = {'LConn', 'RConn'}
    for q = 1:numel(ph.(t{1})), specs{end + 1} = {t{1}, q}; end %#ok<AGROW>
end
close_system(m, 0);
end

function p = lp_in(lib, name, mustContain)
ws = warning('off', 'all'); c = onCleanup(@() warning(ws));
h = find_system(lib, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'Name', name);
h = h(contains(h, mustContain));
if isempty(h), error('pv_ports: block "%s" (%s) not found in %s', name, mustContain, lib); end
p = h{1};
end

% ------------------------------------------------------------------ helpers
function [ok, out] = trial(buildfn)
m = 'pv_ports_tmp';
if bdIsLoaded(m), close_system(m, 0); end
new_system(m);
set_param(m, 'StopTime', '0.01', 'Solver', 'ode23t', 'MaxStep', '1e-3', 'ReturnWorkspaceOutputs', 'on');
try
    buildfn(m);
    out = sim(m);
    ok = true;
catch e
    ok = false; out = e.message;
end
close_system(m, 0);
end

function g = ground(m, L)
r = ab(m, L.ref, 'ref'); s = ab(m, L.solver, 'solver');
g = pt(r, {'LConn', 1});
ln(m, pt(s, {'RConn', 1}), g);
end

function h = ab(m, src, name, varargin)
h = add_block(src, [m '/' name], 'MakeNameUnique', 'on');
if ~isempty(varargin), set_param(h, varargin{:}); end
end

function w = tows(m, var)
w = ab(m, 'simulink/Sinks/To Workspace', var, 'VariableName', var, 'SaveFormat', 'Array');
end

function p = pt(h, spec)
ph = get_param(h, 'PortHandles');
p = ph.(spec{1})(spec{2});
end

function ln(m, a, b)
add_line(m, a, b);
end

function p = lp(lib, name)
ws = warning('off', 'all'); c = onCleanup(@() warning(ws));
h = find_system(lib, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'Name', name);
if isempty(h)
    h = find_system(lib, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'Name', strrep(name, ' ', newline));
end
if isempty(h), error('pv_ports: block "%s" not found in %s', name, lib); end
p = h{1};
end

function s = okmsg(ok, msg)
if ok, s = 'yes'; else, s = ['no (' strtok(strrep(msg, newline, ' '), '.') ')']; end
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end

function p = find_xgb_block()
p = '';
ws = warning('off', 'all'); c = onCleanup(@() warning(ws));
roots = {fullfile(matlabroot, 'toolbox', 'stats'), fullfile(matlabroot, 'toolbox', 'shared')};
for r = 1:numel(roots)
    f = [dir(fullfile(roots{r}, '**', '*lib*.slx')); dir(fullfile(roots{r}, '**', '*lib*.mdl'))];
    for k = 1:numel(f)
        [~, nm] = fileparts(f(k).name);
        try
            load_system(fullfile(f(k).folder, f(k).name));
            h = find_system(nm, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'Name', 'ClassificationXGBoost Predict');
            if ~isempty(h), p = h{1}; return, end
        catch
        end
    end
end
end
