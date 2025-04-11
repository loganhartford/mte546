% Load the .mat file
data = load('D:\Documents\GitHub\mte546\exam\Heart_of_Gold_Improp_drive\Heart_of_Gold_Improp_drive.mat');

% Get field names (variables)
fields = fieldnames(data);

% Loop through each field and write to CSV
for i = 1:numel(fields)
    variable = data.(fields{i});
    
    % Only save numeric arrays
    if isnumeric(variable) || islogical(variable)
        csvwrite([fields{i} '.csv'], variable);
    elseif isstring(variable) || ischar(variable)
        % Save strings as text file
        writematrix(variable, [fields{i} '.csv']);
    else
        disp(['Skipping: ' fields{i} ' (unsupported type)']);
    end
end
