function name = pv_model_name(system, config)
%PV_MODEL_NAME  pv_system1_series, pv_system1_parallel, pv_system2, pv_system3
if system == 1
    name = sprintf('pv_system1_%s', config);
else
    name = sprintf('pv_system%d', system);
end
end
