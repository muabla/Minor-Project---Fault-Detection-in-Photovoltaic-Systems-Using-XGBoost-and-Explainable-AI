function probe_blocks()
%PROBE_BLOCKS  Lists the exact library paths, parameter names and ports of the blocks the PV model needs,
%   in YOUR MATLAB version. Writes everything to probe_output.txt.
%   It only creates a temporary model in memory and closes it; nothing is saved.

here = fileparts(mfilename('fullpath'));
outFile = fullfile(here, 'probe_output.txt');
if exist(outFile, 'file'), delete(outFile); end
diary(outFile); diary on
c = onCleanup(@() diary('off'));
fprintf('=== probe_blocks ===  MATLAB %s\n', version);

libs = {'simulink', 'nesl_utility', 'fl_lib', 'ee_lib', 'statsmllib'};
loaded = {};
for k = 1:numel(libs)
    try
        load_system(libs{k}); loaded{end + 1} = libs{k}; %#ok<AGROW>
        fprintf('library %-14s loaded\n', libs{k});
    catch e
        fprintf('library %-14s NOT FOUND (%s)\n', libs{k}, e.message);
    end
end

wanted = {'Solar Cell', 'Diode', 'Fault', 'Resistor', 'Variable Resistor', 'Controlled Voltage Source', ...
          'Current Sensor', 'Voltage Sensor', 'Electrical Reference', 'Solver Configuration', ...
          'PS-Simulink Converter', 'Simulink-PS Converter', 'PS Constant', 'ClassificationXGBoost Predict'};
found = struct();
fprintf('\n--- where the blocks are ---\n');
for w = 1:numel(wanted)
    hits = {};
    for k = 1:numel(loaded)
        try
            h = find_system(loaded{k}, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'Name', wanted{w});
            hits = [hits; h(:)]; %#ok<AGROW>
        catch
        end
    end
    fprintf('%-30s %s\n', wanted{w}, strjoin(hits', '  |  '));
    found.(matlab.lang.makeValidName(wanted{w})) = hits;
end

mdl = 'pv_probe_tmp';
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
fn = fieldnames(found);
fprintf('\n--- parameters and ports ---\n');
for w = 1:numel(fn)
    hits = found.(fn{w});
    for j = 1:min(3, numel(hits))          % first 3 places each block appears
        src = hits{j};
        try
            b = add_block(src, sprintf('%s/b%d_%d', mdl, w, j), 'MakeNameUnique', 'on');
        catch e
            fprintf('\n[%s] could not add: %s\n', src, e.message); continue
        end
        fprintf('\n[%s]\n', strrep(src, newline, ' '));
        try, fprintf('  MaskType: %s   ReferenceBlock/SourceFile: %s\n', get_param(b, 'MaskType'), get_param(b, 'ReferenceBlock')); catch, end
        try
            dp = get_param(b, 'DialogParameters');
            names = fieldnames(dp);
            for q = 1:numel(names)
                try
                    v = get_param(b, names{q});
                    if ~ischar(v), v = mat2str(v); end
                catch
                    v = '?';
                end
                fprintf('  %-28s = %-22s  "%s"\n', names{q}, v, dp.(names{q}).Prompt);
            end
        catch e
            fprintf('  (no dialog parameters: %s)\n', e.message);
        end
        ph = get_param(b, 'PortHandles');
        pos = get_param(b, 'Position');
        for t = {'Inport', 'Outport', 'LConn', 'RConn'}
            hs = ph.(t{1});
            for q = 1:numel(hs)
                pp = get_param(hs(q), 'Position');
                extra = '';
                try, extra = [' name="' get_param(hs(q), 'Name') '"']; catch, end
                fprintf('  port %-7s %d  at (%+d, %+d) from block centre%s\n', t{1}, q, ...
                    round(pp(1) - (pos(1) + pos(3)) / 2), round(pp(2) - (pos(2) + pos(4)) / 2), extra);
            end
        end
    end
end
close_system(mdl, 0);
fprintf('\n=== done. Output saved to %s ===\n', outFile);
end
