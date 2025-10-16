% Quick test of data parsing
raw_file = '../processedData/Hw02/hw02_qB_100mus/hw02_qB_100mus_0.out';
data = readmatrix(raw_file, 'FileType', 'text', 'Delimiter', ' ', 'ConsecutiveDelimitersRule', 'join');
fprintf('Data size: %d x %d\n', size(data,1), size(data,2));
fprintf('First 3 rows:\n');
disp(data(1:3,:));
