function v = pv_get(name)
%PV_GET  Reads a variable created by setup_pv (pv_info, xgbZero, xgbAdapted, xgbMdl) from the base workspace.
%   If MATLAB was restarted and setup_pv has not run yet in this session, it runs setup_pv first.
if ~evalin('base', sprintf('exist(''%s'', ''var'')', name))
    fprintf('(setup_pv has not run in this MATLAB session - running it now)\n');
    setup_pv();
end
v = evalin('base', name);
end
