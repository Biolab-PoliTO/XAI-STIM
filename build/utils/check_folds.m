clear; close all; clc

all_folds = readtable('all_folds.xlsx');

%%

% fold_assignations = unique(all_folds(:,[2,8,10:27]),'rows');
% writetable(fold_assignations,'fold_assignments.xlsx')

%%

all_folds.class(all_folds.class>1) = 0;
datasets = unique(all_folds.dataset);
num_win_dataset = array2table(zeros(2, length(datasets)),...
                  'VariableNames',datasets,'RowNames',{'out_stn','in_stn'});

for i = 1:length(datasets)
    num_win_dataset.(datasets{i}) = [sum(all_folds.class==0 & strcmp(all_folds.dataset,datasets{i}));
                                     sum(all_folds.class==1 & strcmp(all_folds.dataset,datasets{i}))];
end

bar(1:length(datasets), table2array(num_win_dataset)/1e3, 'stacked')
set(gca,'XTicklabel',datasets), ylabel('# windows (x10^3)'), legend('out-stn','in-stn');


