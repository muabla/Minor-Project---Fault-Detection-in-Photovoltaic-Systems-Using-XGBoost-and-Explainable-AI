function [label, score] = pv_predict_extrinsic(x)
%PV_PREDICT_EXTRINSIC  Used only if the ClassificationXGBoost Predict block is not available:
%   the Raspberry Pi subsystem's MATLAB Function block calls this to run predict() of xgbMdl.
mdl = pv_get('xgbMdl');
[l, s] = predict(mdl, double(x(:)'));
label = pv_label_index(l, mdl) - 1;
score = double(s);
end
