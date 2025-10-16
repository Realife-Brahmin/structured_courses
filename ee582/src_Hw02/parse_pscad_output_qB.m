function data = parse_pscad_output_qB(base_path, file_prefix)
% PARSE_PSCAD_OUTPUT_QB Parse PSCAD .out and .inf files for Circuit B (RL circuit)
%
% Inputs:
%   base_path: Path to the folder containing the files
%   file_prefix: Prefix of the files (e.g., 'hw02_qB_100mus')
%
% Outputs:
%   data: Struct with fields:
%         - t: time vector [s]
%         - e_t: source voltage [V]
%         - vL: inductor voltage [V]
%         - IL: inductor current [A]

    % Read .inf file to get column information
    inf_file = fullfile(base_path, [file_prefix '.inf']);
    fid = fopen(inf_file, 'r');
    if fid == -1
        error('Cannot open .inf file: %s', inf_file);
    end
    
    col_names = {};
    col_units = {};
    col_idx = 1;
    
    while ~feof(fid)
        line = fgetl(fid);
        if contains(line, 'Output')
            % Parse line like: PGB(1) Output Desc="e_t" Group="Main" Max=11 Min=-2 Units = "V"
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
    
    % Use textscan for better handling of scientific notation
    fid2 = fopen(out_file, 'r');
    raw_cell = textscan(fid2, '%f %f %f %f %f %f %f', 'Delimiter', ' ', 'MultipleDelimsAsOne', true);
    fclose(fid2);
    
    % Convert cell array to matrix
    raw_data = [raw_cell{:}];
    fprintf('Read .out file: %d rows x %d columns\n', size(raw_data, 1), size(raw_data, 2));
    
    % Create output struct
    data = struct();
    
    % Based on the .out file format for Circuit B:
    % PSCAD .out files have the first two columns as duplicate time stamps
    % Then the actual data columns follow based on the .inf file
    % Col 1: time (first timestamp)
    % Col 2: time (duplicate, simulation time) <- USE THIS
    % Col 3: e_t (source voltage) - Column 1 from .inf
    % Col 4: vL (inductor voltage) - Column 2 from .inf
    % Col 5: IL (inductor current) - Column 3 from .inf
    % Col 6: BRK (breakpoint marker) - Column 4 from .inf
    
    % Note: readmatrix might add extra columns due to spacing
    % Always use column 2 for time, then skip to actual data columns
    
    data.t = raw_data(:, 1);  % Use first time column (simulation time)
    
    % The actual data starts at column 3 (after two time columns)
    if size(raw_data, 2) >= 3
        data.e_t = raw_data(:, 3);  % Source voltage
    end
    
    if size(raw_data, 2) >= 4
        data.vL = raw_data(:, 4);  % Inductor voltage
    end
    
    if size(raw_data, 2) >= 5
        data.IL = raw_data(:, 5);  % Inductor current
    end
    
    fprintf('\nParsed data structure:\n');
    fprintf('  Time: %d samples from %.3e to %.3e s\n', length(data.t), data.t(1), data.t(end));
    if isfield(data, 'e_t')
        fprintf('  e_t: min=%.3f, max=%.3f V\n', min(data.e_t), max(data.e_t));
    end
    if isfield(data, 'vL')
        fprintf('  vL: min=%.3f, max=%.3f V\n', min(data.vL), max(data.vL));
    end
    if isfield(data, 'IL')
        fprintf('  IL: min=%.3f, max=%.3f A\n', min(data.IL), max(data.IL));
    end
    
end
