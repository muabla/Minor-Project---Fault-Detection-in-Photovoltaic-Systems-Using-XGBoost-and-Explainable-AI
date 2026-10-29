function r = pv_simulate(system, config, scen, ex, doShap)
%PV_SIMULATE  One run of pv_systemN.slx for a scenario: sets the block values (pvp), runs the I-V sweep (0-1 s)
%   and the operating phase (1-1.6 s), and returns what the Raspberry Pi subsystem computed: 22 inputs, class,
%   probabilities, SHAP values, trip; plus the I-V curve and the current over time.
if nargin < 5, doShap = false; end
mdl = pv_model_name(system, config);
here = fileparts(mfilename('fullpath'));
if ~bdIsLoaded(mdl)
    if exist(fullfile(here, [mdl '.slx']), 'file')
        load_system(fullfile(here, [mdl '.slx']));
    else
        build_pv_slx(system, config);
    end
end
pvp = pv_params(system, config, scen, ex);
pvp.doShap = double(doShap);
assignin('base', 'pvp', pvp);
t0 = tic;
out = sim(mdl);
V = all_samples(out.get('pv_V')); I = all_samples(out.get('pv_I'));
x = out.get('pv_x'); lab = out.get('pv_label'); sc = out.get('pv_score');
r.x = last_sample(x); r.labelIdx = last_sample(lab) + 1; r.probs = last_sample(sc);
try, r.shap = last_sample(out.get('pv_shap')); catch, r.shap = zeros(1, 22); end
try, r.trip = last_sample(out.get('pv_trip')); catch, r.trip = 0; end
try, kc = out.get('pv_contactor'); r.contactor = all_samples(kc); catch, r.contactor = []; end
V = V(:); I = I(:);
N = min(1001, numel(V));
sv = 1; if V(N) < 0, sv = -1; end
si = 1; if I(1) < 0, si = -1; end
r.t = (0:numel(I) - 1)' * 1e-3; r.Iall = si * I; r.Vall = sv * V;   % whole run (sweep + operating phase)
r.V = sv * V(1:N); r.I = si * I(1:N);                               % the I-V sweep
[~, raw] = pv_features(r.V, r.I, scen.G, scen.Ta, pvp.maxVoc, pvp.maxIsc, pvp.nStr, pvp.ex);
r.Voc = raw(1); r.Isc = raw(2); r.Vmp = raw(3); r.Imp = raw(4);
r.seconds = toc(t0);
end

function v = last_sample(A)
% value at the last time step. To Workspace (Array) saves a scalar or vector signal as [time x n],
% but a matrix-shaped signal (e.g. 1 x 22) as [1 x 22 x time]
A = double(A);
if ndims(A) == 3
    v = reshape(A(:, :, end), 1, []);
else
    v = A(end, :);
end
end

function v = all_samples(A)
% scalar signal over time as a column vector
A = double(A);
if ndims(A) == 3, v = squeeze(A(1, 1, :)); else, v = A(:, 1); end
v = v(:);
end
