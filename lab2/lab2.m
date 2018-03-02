clear
load irisdata.mat

%% extract unique labels (class names)
labels = unique(irisdata_labels);
numericLabels = zeros(size(irisdata_features,1),1);
for i = 1:size(labels,1)
    numericLabels(find(strcmp(labels{i},irisdata_labels)),:)= i;
end


A = [irisdata_features(1:50,2:3)]; %setosa
B = [irisdata_features(51:100,2:3)]; % versicolour
C = [irisdata_features(101:150,2:3)]; %virginia
label_A = numericLabels(1:50);
label_B = numericLabels(51:100);
label_C = numericLabels(101:150);

%% for question 1
num_of_training = 0.3*50;
trainingSet1 = [A(1:num_of_training,1:2) label_A(1:num_of_training);B(1:num_of_training,1:2) label_B(1:num_of_training)];
a_set1 = training(trainingSet1,0.01,300,[0 0 1]',0)

figure(1);
scatter(trainingSet1(:,1), trainingSet1(:,2));
title("Training Set 1");
xlabel("X1");
ylabel("X2");
hold on;
plot(trainingSet1(:,1), (-1*a_set1(2)/a_set1(3))*trainingSet1(:,1) - a_set1(1)/a_set1(3))
hold off;

%% for question 2
num_of_training = 0.3*50;
testingSet1 = [A(num_of_training+1:50,1:2) label_A(num_of_training+1:50);B(1+num_of_training:50,1:2) label_B(num_of_training+1:50)];
errorCalc(a_set1,testingSet1)

figure(2);
scatter(testingSet1(:,1), testingSet1(:,2));
title("Testing Set 1");
xlabel("X1");
ylabel("X2");
hold on;
plot(testingSet1(:,1), (-1*a_set1(2)/a_set1(3))*testingSet1(:,1) - a_set1(1)/a_set1(3))
hold off;

%% for question 3
num_of_training = 0.7*50;
trainingSet1_flip = [A(1:num_of_training,1:2) label_A(1:num_of_training);B(1:num_of_training,1:2) label_B(1:num_of_training)];
testingSet1_flip = [A(num_of_training+1:50,1:2) label_A(num_of_training+1:50);B(1+num_of_training:50,1:2) label_B(num_of_training+1:50)];

a_set1_flip = training(trainingSet1_flip,0.01,300,[0 0 1]',0)
errorCalc(a_set1_flip,testingSet1_flip)
pause;

figure(3);
scatter(trainingSet1_flip(:,1), trainingSet1_flip(:,2));
title("Training Set 1 Flipped");
xlabel("X1");
ylabel("X2");
hold on;
plot(trainingSet1_flip(:,1), (-1*a_set1_flip(2)/a_set1_flip(3))*trainingSet1_flip(:,1) - a_set1_flip(1)/a_set1_flip(3))
hold off;

figure(4);
scatter(testingSet1_flip(:,1), testingSet1_flip(:,2));
title("Testing Set 1 Flipped");
xlabel("X1");
ylabel("X2");
hold on;
plot(testingSet1_flip(:,1), (-1*a_set1_flip(2)/a_set1_flip(3))*testingSet1_flip(:,1) - a_set1_flip(1)/a_set1_flip(3))
hold off;

%% for question 4
num_of_training = 0.3*50;
trainingSet2 = [B(1:num_of_training,1:2) label_B(1:num_of_training);C(1:num_of_training,1:2) label_C(1:num_of_training)];
testingSet2 = [B(num_of_training+1:50,1:2) label_B(num_of_training+1:50);C(1+num_of_training:50,1:2) label_C(num_of_training+1:50)];

a_set2= training(trainingSet2,0.01,300,[0 0 1]',0)
errorCalc(a_set2,testingSet2)
pause;
num_of_training = 0.7*50;
trainingSet2_flip = [B(1:num_of_training,1:2) label_B(1:num_of_training);C(1:num_of_training,1:2) label_C(1:num_of_training)];
testingSet2_flip = [B(num_of_training+1:50,1:2) label_B(num_of_training+1:50);C(1+num_of_training:50,1:2) label_C(num_of_training+1:50)];

a_set2_flip = training(trainingSet2_flip,0.01,300,[0 0 1]',0)
errorCalc(a_set2_flip,testingSet2_flip)

figure(5);
scatter(trainingSet2(:,1), trainingSet2(:,2));
title("Training Set 2");
xlabel("X1");
ylabel("X2");
hold on;
plot(trainingSet2(:,1), (-1*a_set2(2)/a_set2(3))*trainingSet2(:,1) - a_set2(1)/a_set2(3))
hold off;

figure(6);
scatter(testingSet2(:,1), testingSet2(:,2));
title("Testing Set 2");
xlabel("X1");
ylabel("X2");
hold on;
plot(testingSet2(:,1), (-1*a_set2(2)/a_set2(3))*testingSet2(:,1) - a_set2(1)/a_set2(3))
hold off;

figure(7);
scatter(trainingSet2_flip(:,1), trainingSet2_flip(:,2));
title("Training Set 2 Flipped");
xlabel("X1");
ylabel("X2");
hold on;
plot(trainingSet2_flip(:,1), (-1*a_set2_flip(2)/a_set2_flip(3))*trainingSet2_flip(:,1) - a_set2_flip(1)/a_set2_flip(3))
hold off;

figure(8);
scatter(testingSet2_flip(:,1), testingSet2_flip(:,2));
title("Testing Set 2 Flipped");
xlabel("X1");
ylabel("X2");
hold on;
plot(testingSet2_flip(:,1), (-1*a_set2_flip(2)/a_set2_flip(3))*testingSet2_flip(:,1) - a_set2_flip(1)/a_set2_flip(3))
hold off;

%% for question 5
LR1 = 0.01;
LR2 = 2;
a1 = [0 0 1]';
a2 = [0 1 0]';
itr = 300;
theta = 0;

a_set_LR1 = training(trainingSet1, LR1, itr, a1, theta)
a_set_LR2 = training(trainingSet1, LR2, itr, a1, theta)

a_set_a1 = training(trainingSet1, LR1, itr, a1, theta)
a_set_a2 = training(trainingSet1, LR1, itr, a2, theta)

figure(9);
scatter(trainingSet1(:,1), trainingSet1(:,2));
title("Training Set 1, LR = 2");
xlabel("X1");
ylabel("X2");
hold on;
plot(trainingSet1(:,1), (-1*a_set_LR2(2)/a_set_LR2(3))*trainingSet1(:,1) - a_set_LR2(1)/a_set_LR2(3))
hold off;

figure(10);
scatter(trainingSet1(:,1), trainingSet1(:,2));
title("Training Set 1, a2 = [0 1 0]");
xlabel("X1");
ylabel("X2");
hold on;
plot(trainingSet1(:,1), (-1*a_set_a2(2)/a_set_a2(3))*trainingSet1(:,1) - a_set_LR2(1)/a_set_a2(3))
hold off;