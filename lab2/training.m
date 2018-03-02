function [a] = training(trainingSet,learningRate,num_of_iters,a,theta)
    %using perceptron criterion and gradient descent
	augmnt = [ones(size(trainingSet,1),1) trainingSet(:,(1:2))]'; %augmentation
    size_half = size(augmnt,2)/2; 
    normalize = [augmnt(:,1:size_half) -augmnt(:,size_half+1:2*size_half)]; %normalization
    
    counter = 0;
        while (true)
             counter= counter + 1;
             [row,column] = find(a'*normalize<=0);
             temp = normalize(:,column);
             delta_J = -sum(temp,2); %the gradient
             condition = learningRate*delta_J;
             a = a - condition;
            if (counter >= num_of_iters) || (all(abs(condition) <= theta))
                break;
            end
        end
    disp(['took ',num2str(counter), ' iterations'])
    end