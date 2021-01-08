function [bags, num_data, num_feature] = MIL_Data_Load(filename)

global preprocess;

matrix_file = [filename '.matrix'];
label_file  = [filename '.label'];
insts = [];

if preprocess.InputFormat == 1
    %sparse input format
    strcmd = sprintf('!ReadInput.pl %s 1', filename);
    eval(strcmd);
    D = load(matrix_file);
    insts = spconvert(D);
else
    strcmd = sprintf('!ReadInput.pl %s', filename);
    eval(strcmd);    
    insts = load(matrix_file);
end
        
fid = fopen(label_file, 'r');
if fid == -1, error('The label file is not generated, quitting...'); end;

nbag = 0;
prev_bag_name = '';
ninst = 0;
idx = 0;
while feof(fid) == 0

    line = strtrim(fgets(fid));
    %elems = strsplit(' ',line);    %instance_name, bag_name, label
    elems = strsplit(line); 
    
    bag_name = cell2mat(elems(2));

    if strcmp(bag_name, prev_bag_name) == 0     %change of bag
        if (nbag >= 1)
            bags(nbag).instance = insts(idx + 1:idx + ninst, :);
            bags(nbag).label = any(bags(nbag).inst_label);
            idx = idx + ninst;
        end;
        nbag = nbag + 1;
        bags(nbag).name = bag_name;
        %tang新加的代码
        %fprintf("%d:%s\n",nbag,bags(nbag).name);
       
        prev_bag_name = bag_name;
        ninst = 0;
    end
    
    ninst = ninst + 1;
    bags(nbag).inst_name(ninst) = elems(1);
    label = cell2mat(elems(3));
    
    bags(nbag).inst_label(ninst) = strcmp(label,'1');   %the positive label must be set to 1
    

end;

if (nbag >= 1)
    bags(nbag).instance = insts(idx + 1:idx + ninst,:);
    bags(nbag).label = any(bags(nbag).inst_label);
end;
fclose(fid);

num_data = length(bags);
num_feature = size(bags(1).instance, 2);


% normalize the data set
if (preprocess.Normalization == 1) 
    bags = MIL_Scale(bags);
end;

% randomize the data
rand('state',sum(100*clock));
if (preprocess.Shuffled == 1) %Shuffle the datasets
    Vec_rand = rand(num_data, 1);
    [B, Index] = sort(Vec_rand);
    bags = bags(Index);
end;

