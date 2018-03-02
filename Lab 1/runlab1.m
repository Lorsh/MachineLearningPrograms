%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAB 1, Bayesian Decision Theory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Attribute Information for IRIS data:
%    1. sepal length in cm
%    2. sepal width in cm
%    3. petal length in cm
%    4. petal width in cm

%    class label/numeric label: 
%       -- Iris Setosa / 1 
%       -- Iris Versicolour / 2
%       -- Iris Virginica / 3


%% this script will run lab1 experiments..
clear
load irisdata.mat

%% extract unique labels (class names)
labels = unique(irisdata_labels);

%% generate numeric labels
numericLabels = zeros(size(irisdata_features,1),1);
for i = 1:size(labels,1)
    numericLabels(find(strcmp(labels{i},irisdata_labels)),:)= i;
end

%% feature distribution of x1 for two classes
% figure
% 
%     
% subplot(1,2,1), hist(irisdata_features(find(numericLabels(:)==1),2),100), title('Iris Setosa, sepal width (cm)');
% subplot(1,2,2), hist(irisdata_features(find(numericLabels(:)==2),2),100); title('Iris Veriscolour, sepal width (cm)');
% 
% figure
% 
% subplot(1,2,1), hist(irisdata_features(find(numericLabels(:)==1),1),100), title('Iris Setosa, sepal length (cm)');
% subplot(1,2,2), hist(irisdata_features(find(numericLabels(:)==2),1),100); title('Iris Veriscolour, sepal length (cm)');
%     
% 
% figure
% 
% plot(irisdata_features(find(numericLabels(:)==1),1),irisdata_features(find(numericLabels(:)==1),2),'rs'); title('x_1 vs x_2');
% hold on;
% plot(irisdata_features(find(numericLabels(:)==2),1),irisdata_features(find(numericLabels(:)==2),2),'k.');
% axis([4 7 1 5]);
% hold off
    

%% build training data set for two class comparison
% merge feature samples with numeric labels for two class comparison (Iris
% Setosa vs. Iris Veriscolour
trainingSet = [irisdata_features(1:100,:) numericLabels(1:100,1) ];



%% Lab1 experiments (include here)
%Mean & Std for class 1 & 2
% mean:3.418
% std:0.38102
% mean:2.77
% std:0.3138
%%3)
result1 = lab1(3.3,trainingSet,2);
result2 = lab1(4.4,trainingSet,2);
result3 = lab1(5,trainingSet,2);
result4 = lab1(5.7,trainingSet,2);
result5 = lab1(6.3,trainingSet,2);


disp(['x=3.3: ',num2str(result1)])
disp(['x=4.4: ',num2str(result2)])
disp(['x=5: ',num2str(result3)])
disp(['x=5.7: ',num2str(result4)])
disp(['x=6.3: ',num2str(result5)])

pause;
%%4)To find the optimal threshold, we plot the two gaussian distributions
%%that are associated with each feature for the two classes, and find the
%%the intersection.

gauss_pdf = zeros(length([0:0.01:10]),2);
gx_vec = zeros(length([0:0.01:10]),1);
counter = 1;
for i=0:.01:10
    [~,gx_vec(counter),gauss_pdf(counter,:)] = lab1(i,trainingSet,2);
    counter = counter +1;
end
figure(4)
plot([0:0.01:10],gauss_pdf(:,1))
hold on
plot([0:0.01:10],gauss_pdf(:,2),'--')
plot([0:0.01:10],gx_vec,':')
xlabel('Sepal Width')
ylabel('Gaussian distribution')
legend('Setosa','Versicolour','Discriminant Function')
title('P(X|w) for Sepal Width')
grid on
hold off
backup = gx_vec;


gauss_pdf = zeros(length([0:0.01:10]),2);
gx_vec = zeros(length([0:0.01:10]),1);
counter = 1;
for i=0:.01:10
    [~,gx_vec(counter),gauss_pdf(counter,:)] = lab1(i,trainingSet,1);
    counter = counter +1;
end
figure(5)
plot([0:0.01:10],gauss_pdf(:,1))
hold on
plot([0:0.01:10],gauss_pdf(:,2),'--')
plot([0:0.01:10],gx_vec,':')
xlabel('Sepal Length')
ylabel('Gaussian distribution')
legend('Setosa','Versicolour','Discriminant Function')
title('P(X|w) for Sepal Length')
grid on
hold off


pause;
%%5) If the cost associated with labeling versicolour as setosa is higher
%%than the other way around, we can assign a scale factor associated with
%%the cost of making this error on the posterior probability of
%%versicolour. The threshold will then change as a result, and it can be
%%seen in the discriminant function shifting

gauss_pdf = zeros(length([0:0.01:10]),2);
gx_vec = zeros(length([0:0.01:10]),1);
counter = 1;
for i=0:.01:10
    [~,gx_vec(counter),gauss_pdf(counter,:)] = lab1(i,trainingSet,2,5);
    counter = counter +1;
end
figure(6)
subplot(2,1,1)
plot([0:0.01:10],gauss_pdf(:,1))
hold on
plot([0:0.01:10],gauss_pdf(:,2),'--')
plot([0:0.01:10],gx_vec,'-.')
xlabel('Sepal Width')
ylabel('Gaussian distribution')
legend('Setosa','Versicolour','Discriminant Function')
title('P(X|w) for Sepal Width without modified cost')
axis([0 10 -1.5 1.5])
grid on;
subplot(2,1,2)
plot([0:0.01:10],gauss_pdf(:,1))
hold on
plot([0:0.01:10],gauss_pdf(:,2),'--')
plot([0:0.01:10],backup,'-.')
xlabel('Sepal Width')
ylabel('Gaussian distribution')
title('P(X|w) for Sepal Width with modified cost')
axis([0 10 -1.5 1.5])
grid on
hold off

pause;

%%6) To determine which feature is better, we can calculate the area
%%underneath the intersection of the two gaussian pdfs. That area is the
%%most error prone region of the discriminator. The feature that produces
%%the smallest region of intersection is the one that is better for
%%catagorizing the samples.

calc_overlap_twonormal(0.38102,0.3138,3.418,2.77,1,5,0.01);  %%area under the curves for sepal width
pause;
calc_overlap_twonormal(0.353,0.516,5,5.936,0,10,0.01);  %%area under the curves for sepal length