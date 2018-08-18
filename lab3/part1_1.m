%% Run Back Propagation tests
%  This code assumes there is one output neuron per class!

clc;
close all
clear; 

%% test 1 - XOR data
x1 =[1,1,0,0] %XOR 
x2 = [1,0,1,0]%~XOR
targets =[0,1,1,0] ; % 
patterns =[1 1 1 1;x1 ;x2] ; %


%% XOR tests
netXOR = backprop(patterns,targets,2,1,0.1,0.001); %e.g. backprop(patterns, targets, nH, bias, eta, theta)


WineData = load('wine.data');

TrainingSet=zeros(107,3);
TrainingSet(1:59,:,:)= WineData(1:59,1:3); %Class 1=w1
TrainingSet(60:107,:,:)=WineData(131:178,1:3); %Class 3=w2

for i=60:length(TrainingSet)
    TrainingSet(i,1)=-1;
end

t = TrainingSet(:,1)'; % Target should be the actual class labels (ie seperation of classes) 
x1 = TrainingSet(:,2)'; % Input of alchol content 
x2 = TrainingSet(:,3)'; % Input of Malic Acid content
for q=1:length(x1)
x1n(q)=(x1(q)-mean(x1))/(std(x1)); 
x2n(q)=(x2(q)-mean(x2))/(std(x2)); 
end

patterns =[(1:107);x1n ;x2n] ; 
netWine = backprop(patterns,t,2,1,0.02,0.01); %e.g. backprop(patterns, targets, nH, bias, eta, theta)

