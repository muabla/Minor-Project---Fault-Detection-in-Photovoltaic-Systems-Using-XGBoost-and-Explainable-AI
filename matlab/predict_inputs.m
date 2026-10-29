function predict_inputs(x, which)
%PREDICT_INPUTS  Backup check: the 22 inputs copied from the website -> the same trained model -> prediction.
%   predict_inputs([0.92 0.81 ...])              library model (zero-shot)
%   predict_inputs([...], 'adapted1')            model after 5 readings per class of System 1 (adapted2, adapted3)
if nargin < 2, which = 'zero'; end
info = pv_get('pv_info');
if strcmp(which, 'zero')
    mdl = pv_get('xgbZero');
else
    A = pv_get('xgbAdapted'); mdl = A{str2double(which(end))};
end
x = double(x(:)');
assert(numel(x) == 22, 'Expected 22 inputs, got %d', numel(x));
[lab, score] = predict(mdl, x);
names = pv_class_names(mdl, info);
fprintf('\nModel: %s\nPrediction: %s\n\n', which, upper(names{pv_label_index(lab, mdl)}));
for k = 1:numel(names)
    fprintf('  %-16s %6.1f %%  %s\n', names{k}, 100 * score(k), repmat('#', 1, round(30 * score(k))));
end
fprintf('\nInputs used:\n');
for k = 1:22
    fprintf('  %-11s %9.5f\n', info.feature_order{k}, x(k));
end
assignin('base', 'pvlast_inputs', struct('inputs', x, 'label', names{pv_label_index(lab, mdl)}, 'probs', score));
end
