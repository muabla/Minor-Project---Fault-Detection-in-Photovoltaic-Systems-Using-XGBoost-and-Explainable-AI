function phi = pv_shap_extrinsic(x, label)
%PV_SHAP_EXTRINSIC  SHAP values of the 22 inputs for the predicted class (called by the "SHAP explanation" block
%   inside the Raspberry Pi subsystem when the sweep is finished). Uses MATLAB's shapley() on the same imported
%   XGBoost model, with 120 training-library readings as the reference ("background") data.
%   phi(k) > 0 : input k pushed the model TOWARD the predicted class;  < 0 : away from it.
mdl = pv_get('xgbMdl'); bg = pv_get('pv_bg');
x = double(x(:)');
try
    e = shapley(mdl, bg, 'QueryPoints', x);
catch
    e = shapley(mdl, bg, 'QueryPoint', x);
end
S = e.ShapleyValues;
V = S{:, 2:end};                         % 22 x 6 (one column per class, in mdl.ClassNames order)
if iscell(V), V = cell2mat(V); end
col = double(label) + 1;
if size(V, 2) < col, col = size(V, 2); end
phi = double(V(:, col))';
end
