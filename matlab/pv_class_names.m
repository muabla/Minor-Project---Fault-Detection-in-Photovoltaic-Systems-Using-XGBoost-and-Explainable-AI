function names = pv_class_names(mdl, info)
%PV_CLASS_NAMES  Class names in the order of the model's score columns.
cn = mdl.ClassNames;
names = cell(1, numel(cn));
for k = 1:numel(cn)
    names{k} = info.classes{pv_label_index(cn(k), mdl)};
end
end
