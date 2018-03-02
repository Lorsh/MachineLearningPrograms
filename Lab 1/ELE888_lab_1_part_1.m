count_S=0;
count_V=0;
val=3.1000;
count=0;
% Loop through the training set
for i=1:100;
    if val==trainingSet(i,2)
        count=count +1;
        %check if it is a Setosa
        if trainingSet(i,5)==1
            count_S=count_S+1;
        end
        %check if it is a Versicolour
        if trainingSet(i,5)==2
            count_V=count_V +1;
        end
    end
end
% Joint probablity of Setosa and Versicolour
jointP_S_Val= count_S/count;
jointP_V_Val= count_V/count;

% Condtional probablity of Setosa and Versicolour guess
cond_P_S= jointP_S_Val/0.5;
cond_P_V= jointP_V_Val/0.5;

% Probablity of the guess
prob_P=(cond_P_V+ coV+ cond_P_S)*0.5

% Final condition
g1=(cond_P_S *.5 )/prob_P
g2=(cond_P_V *.5 )/prob_P
if g1-g2>0
   disp('Setosa!!!:')
else
    disp('Versicolour!!')
end

