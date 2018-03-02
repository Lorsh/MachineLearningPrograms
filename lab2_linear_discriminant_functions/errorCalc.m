function [percent] = errorCalc(a,x)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
half_size = size(x,1)/2;
augmnt = [ones(size(x,1),1) x(:,(1:2))]';
g_x = a'*augmnt;
result = [g_x(1:half_size) > 0, g_x(half_size+1:2*half_size) <= 0];
correct= sum(result);
percent = correct/(half_size*2)*100;

disp(['accuracy is ',num2str(percent), '%'])

end

