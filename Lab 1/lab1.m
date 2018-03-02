%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ELE 888/ EE 8209: LAB 1: Bayesian Decision Theory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [posteriors_x,g_x,cp_comb]=lab1(x,Training_Data,feature,cost)

% x = individual sample to be tested (to identify its probable class label)
% feature = index of relevant feature (column) in Training_Data 
% Train_Data = Matrix containing the training samples and numeric class labels
% posteriors_x  = Posterior probabilities
% g_x = value of the discriminant function
% cp_comb = vector gaussian values for given X
% cost is the cost associated with labelling w2 as w1 (versicolour as
% setosa)

if nargin == 3
    cost = 1;
end


D=Training_Data;

% D is MxN (M samples, N columns = N-1 features + 1 label)
[M,N]=size(D);    
 
f=D(:,feature);  % feature samples
la=D(:,N); % class labels


%% %%%%Prior Probabilities%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hint: use the commands "find" and "length"

disp('Prior probabilities: ');
Pr1 = length(find(la < 2)) /  length(la);
Pr2 = length(find(la == 2)) /  length(la);

%% %%%%%Class-conditional probabilities%%%%%%%%%%%%%%%%%%%%%%%

disp('Mean & Std for class 1 & 2');
f1 = f(find(la == 1));
m11 = mean(f1); % mean of the class conditional density p(x/w1)
std11 = std(f1); % Standard deviation of the class conditional density p(x/w1)

disp(['mean:',num2str(m11)]) 
disp(['std:',num2str(std11)]) 
f2 = f(find(la == 2));
m12 = mean(f2);% mean of the class conditional density p(x/w2)
std12= std(f2); % Standard deviation of the class conditional density p(x/w2)

disp(['mean:',num2str(m12)]) 
disp(['std:',num2str(std12)])

disp(['Conditional probabilities for x=' num2str(x)]);
cp11= normpdf(x, m11, std11); % use the above mean, std and the test feature to calculate p(x/w1)

cp12= normpdf(x, m12, std12); % use the above mean, std and the test feature to calculate p(x/w2)

cp_comb = [cp11,cp12];
p_x = cp11*Pr1 + cp12*Pr2;
%% %%%%%%Compute the posterior probabilities%%%%%%%%%%%%%%%%%%%%

%disp('Posterior prob. for the test feature');

pos11= cp11*Pr1/p_x; % p(w1/x) for the given test feature value

pos12= cp12*Pr2/p_x; % p(w2/x) for the given test feature value

posteriors_x = [pos11,pos12];

disp(posteriors_x);
%% %%%%%%Discriminant function for min error rate classifier%%%

disp('Discriminant function for the test feature');

g_x= cost*pos11 - pos12 ; % compute the g(x) for min err rate classifier.
disp(g_x);

