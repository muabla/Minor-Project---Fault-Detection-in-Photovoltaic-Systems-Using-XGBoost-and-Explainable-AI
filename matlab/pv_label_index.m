function m = pv_label_index(label, mdl)
%PV_LABEL_INDEX  1-based class index (1 = normal ... 6 = degradation) for a label returned by predict.
%   Works whether the imported model's class names are numbers (0..5), strings or categories.
if isnumeric(label) || islogical(label)
    m = double(label) + 1;
else
    m = str2double(string(label)) + 1;
end
if isnan(m) || m < 1 || m > 6
    error('pv_label_index:unknown', 'Unexpected class label %s (class names: %s)', string(label), mat2str(string(mdl.ClassNames(:)')));
end
end
