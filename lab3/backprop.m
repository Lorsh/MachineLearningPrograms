function result = backprop(patterns, targets, nH, bias, eta, theta)


%e.g. backprop(patterns, targets, nH, bias, eta, theta)

% patterns = training set / test set of input patterns
% targets = training /test set of output patterns
% eta = learning rate
% theta = threshold for stop criterion
%
% network = MNN model, with trained weights etc:
%   network.nI = number input nodes
%   network.nH = number hidden nodes
%   network.nO = number output nodes
%   network.WH = input to hidden weights (includes bias as nH+1 term)
%   network.WO = hidden to output weights (includes bias as nO+1 term)


%% Init network topology & params
nI = 2; %
nO = 1; %
nS = size(patterns,2); % number of training samples

epoch = 0; %
rate = theta+1; %

figure(1), title('J(avg) vs iteration');


%% setup network weights and init
Wjk = [-1 0.7 -0.4]; %hidden to output weights [-1/sqrt(nH) < Wkj < +1/sqrt(nH)]
Wij = [0.5 1 1;-1.5 1 1]; %input to hidden weights  [-1/sqrt(nI) < Wkj < +1/sqrt(nI)]


%% Calculate total average error
Jp  = 0;
for m = 1:nS
    
    % 1. get next input & target vector
    xi = patterns(:,m); %
    tk = targets(m); %
    
    % 2. propagate input to give output zk
    netj = Wij*xi; %
    [yj, dfnetj] = activation(netj); 
    yj = [1;yj];
    netk = Wjk*yj; %
    [zk, dfnetk] = activation(netk);
    
    Jp = (zk-tk)^2 + Jp; % update Jp
end
J = norm(Jp)*0.5; %


%% for each epoch
while ((rate > theta) && (epoch<10000))
    
    % reset accumulators
       dWOkj = 0;
       dWHij = 0;
    % for each pattern/target pair (one full pass through = one epoch)
    for m = 1:nS
               
        % 1. get next input & target vector              
        xi = patterns(:,m); %
        tk = targets(m); %
        
        % 2. propagate input to give output zk  
        netj = Wij*xi; %
        [yj, df_yj] = activation(netj); 
        yj = [1;yj];  %output of hidden layer
        netk = Wjk*yj; %
        [zk, df_zk] = activation(netk);
        
        % 3. find sensitivity of k & change in WOkj  
        deltak = (tk-zk)*df_zk; % ".*" does for each k
        dWOkj = eta*yj*deltak' + dWOkj; %
        
        % 4. find sensitivity of j & changes in Wij       
        deltaj = df_yj.*(Wjk(2:3)*deltak)'; %
        dWHij = eta*deltaj*xi' + dWHij; %
    end
    
    % update weights using accumulated changes over previous epoch
    Wjk = Wjk + dWOkj'; %
    Wij = Wij + dWHij; %
    epoch = epoch + 1; %
    
    
    % Calculate total average error (over all patterns)
    Jp  = 0;
    for m = 1:nS

        % 1. get next input & target vector
        xi = patterns(:,m); %
        tk = targets(m); %

        % 2. propagate input to give output zk
        netj = Wij*xi; %
        [yj, dfnetj] = activation(netj);
        yj = [1;yj];
        netk = Wjk*yj; %
        [zk(m), dfnetk] = activation(netk);

        Jp = (zk(m)-tk).^2 + Jp; % update Jp
    end
    J(epoch) = norm(Jp)*0.5; % build array of J values each epoch
    if (epoch == 1)
        rate = J(epoch);
    else
        rate = abs(J(epoch-1) - J(epoch)); % rate of change in J
    end
    
    
    
    % display track error plot
    if mod(epoch,1)==0
        figure(1), plot(J); % track errors as plot
        title('J(avg) vs epoch');
    end
    
end
disp(['# of epochs needed for convergence: ',num2str(epoch)]);

%% save final results of trained network & associated classification
%Find the accuracy
correct=0;
accuracy=0;
for i=1:(length(patterns(:,1))-1)
    if floor(zk(i))==targets(i) || ceil(zk(i))==targets(i)
        correct=correct+1;
    end
end
accuracy=correct*100/(length(patterns(:,1))-1);
disp('Accuracy');
accuracy

result = [Wij ; Wjk];
end
function [f, df] = activation(x)
% x = net ... can be single value or vector (if many nH or many nO)
    f = tanh(x);
    df = sech(x).^2;
end
