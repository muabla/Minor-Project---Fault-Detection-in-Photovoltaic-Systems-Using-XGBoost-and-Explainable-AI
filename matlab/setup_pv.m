function setup_pv()
%SETUP_PV  Run once per MATLAB session, from inside the "matlab" folder.
%   - imports the trained XGBoost models (trained in Python, saved as JSON)
%   - loads the PV system data exported from Python (pv_data/systems.json)
%   - checks that MATLAB gives exactly Python's predictions on 18 known readings
%
%   Creates in the base workspace:
%     xgbZero      library model (zero-shot)            xgbAdapted{1..3}  after 5 readings per class
%     xgbMdl       the model the Raspberry Pi block uses (starts as xgbZero)
%     pv_info      systems, classes, feature order, constants

here = fileparts(mfilename('fullpath'));
addpath(here);
D = fullfile(here, 'pv_data');
fprintf('\n=== setup_pv ===  MATLAB %s\n', version);

info = jsondecode(fileread(fullfile(D, 'systems.json')));
assignin('base', 'pv_info', info);
fprintf('Systems loaded: %s\n', strjoin(fieldnames(info.systems)', ', '));

% ---- import the trained models
xgbZero = importModelFromXGBoost(fullfile(D, 'xgb_source.json'));
xgbAdapted = cell(1, 3);
for s = 1:3
    xgbAdapted{s} = importModelFromXGBoost(fullfile(D, sprintf('xgb_adapted_system%d.json', s)));
end
assignin('base', 'xgbZero', xgbZero);
assignin('base', 'xgbAdapted', xgbAdapted);
assignin('base', 'xgbMdl', xgbZero);
B = readtable(fullfile(D, 'shap_background.csv'), 'VariableNamingRule', 'preserve');
assignin('base', 'pv_bg', B{:, info.feature_order'});        % reference readings for SHAP (shapley)
fprintf('Model class: %s   classes: %s   predictors: %d\n', class(xgbZero), ...
    mat2str(xgbZero.ClassNames(:)'), numel(xgbZero.PredictorNames));

% ---- check: 18 known readings, MATLAB prediction vs Python prediction
T = readtable(fullfile(D, 'test_vectors.csv'), 'VariableNamingRule', 'preserve');
X = T{:, info.feature_order'};
[lab, score] = predict(xgbZero, X);
names = pv_class_names(xgbZero, info);
pyP = T{:, {'py_p0', 'py_p1', 'py_p2', 'py_p3', 'py_p4', 'py_p5'}};
ok = 0;
fprintf('\n%-4s %-17s %-17s %-17s %s\n', 'sys', 'true class', 'Python predicts', 'MATLAB predicts', 'max prob. difference');
for k = 1:height(T)
    m = pv_label_index(lab(k), xgbZero);
    d = max(abs(score(k, :) - pyP(k, :)));
    same = (m - 1) == T.py_pred(k);
    ok = ok + same;
    fprintf('%-4d %-17s %-17s %-17s %.2e %s\n', T.system(k), info.classes{T.true_class(k) + 1}, ...
        info.classes{T.py_pred(k) + 1}, names{m}, d, ternary(same, '', '  <-- DIFFERENT'));
end
fprintf('\nSame prediction as Python: %d of %d\n', ok, height(T));
if ok == height(T)
    fprintf('SETUP OK. The trained model works in MATLAB.\n\n');
else
    fprintf('SETUP PROBLEM: send this output to Claude.\n\n');
end
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end
