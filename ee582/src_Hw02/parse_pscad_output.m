function data = parse_pscad_output(base_path, file_prefix)
% PARSE_PSCAD_OUTPUT Parse PSCAD .out and .inf files
%
% Inputs:
%   base_path: Path to the folder containing the files
%   file_prefix: Prefix of the files (e.g., 'hw02_qA_100mus')
%
% Outputs:
%   data: Struct with fields based on .inf file
%         - t: time vector
%         - vC: capacitor voltage
%         - IC: capacitor current
%         - etc.

    % Read .inf file to get column information
    inf_file = fullfile(base_path, [file_prefix '.inf']);
    fid = fopen(inf_file, 'r');
    if fid == -1
        error('Cannot open .inf file: %s', inf_file);
    end
    
    col_names = {};
    col_desc = {};
    col_units = {};
    col_idx = 1;
    
    while ~feof(fid)
        line = fgetl(fid);
        if contains(line, 'Output')
            % Parse line like: PGB(1) Output Desc="e_t" Group="Main" Max=12 Min=8 Units = "V"
            tokens = regexp(line, 'Desc="([^"]*)".*Units\s*=\s*"([^"]*)"', 'tokens');
            if ~isempty(tokens)
                col_names{col_idx} = tokens{1}{1};
                col_units{col_idx} = tokens{1}{2};
                col_idx = col_idx + 1;
            end
        end
    end
    fclose(fid);
    
    fprintf('Found %d columns in .inf file:\n', length(col_names));
    for i = 1:length(col_names)
        fprintf('  Col %d: %s [%s]\n', i, col_names{i}, col_units{i});
    end
    
    % Read .out file (numerical data)
    out_file = fullfile(base_path, [file_prefix '_0.out']);
    if ~exist(out_file, 'file')
        error('Cannot find .out file: %s', out_file);
    end
    
    raw_data = readmatrix(out_file, 'FileType', 'text', 'Delimiter', ' ', 'ConsecutiveDelimitersRule', 'join');
    fprintf('Read .out file: %d rows x %d columns\n', size(raw_data, 1), size(raw_data, 2));
    
    % Create output struct
    data = struct();
    
    % Based on the .out file format:
    % Col 1: time (first timestamp)
    % Col 2: time (duplicate, simulation time)
    % Col 3: e_t (source voltage)
    % Col 4: vC (capacitor voltage)
    % Col 5: IC (capacitor current)
    % Col 6: BRK (breakpoint marker)
    
    data.t = raw_data(:, 2);  % Use second time column (simulation time)
    
    if size(raw_data, 2) >= 4
        data.v_source = raw_data(:, 4);  % Source voltage (e_t)
    end
    
    if size(raw_data, 2) >= 5
        data.vC = raw_data(:, 5);  % Capacitor voltage
    end
    
    if size(raw_data, 2) >= 6
        data.IC = raw_data(:, 6);  % Capacitor current
    end
    
    fprintf('\nParsed data structure:\n');
    fprintf('  Time: %d samples from %.3e to %.3e s\n', length(data.t), data.t(1), data.t(end));
    if isfield(data, 'vC')
        fprintf('  vC: min=%.3f, max=%.3f V\n', min(data.vC), max(data.vC));
    end
    if isfield(data, 'IC')
        fprintf('  IC: min=%.3f, max=%.3f A\n', min(data.IC), max(data.IC));
    end
    
end
