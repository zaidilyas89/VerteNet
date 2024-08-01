clear all
close all
clc

folder1 = uigetdir();
fileList1 = dir(fullfile(folder1, '*.jpg'));

Intended_X_Dim = 1024;
Intended_Y_Dim = 512; 

for i = 1:size(fileList1,1)
    a = imread(fileList1(i).name);
    x_dim = size(a,1);
    y_dim = size(a,2);
    x_ratio = Intended_X_Dim/x_dim;
    y_ratio = Intended_Y_Dim/y_dim;
    factor(i,:) = [x_dim,y_dim,x_ratio,y_ratio];
end

save('factor.mat','factor')
    
