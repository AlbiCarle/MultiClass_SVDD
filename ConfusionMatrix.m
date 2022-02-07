function [] = ConfusionMatrix(y, y_pred, Num_class)

% Confusion Matrix
% Usage: ConfusionMatrix(y, y_pred, Num_class)

CM = zeros(Num_class, Num_class);

m = [y y_pred];

for i = 1:Num_class
    for j = 1:Num_class
        
        CM(i,j) = sum(m(:,1)==i & m(:,2)==j);

    end
end    

disp('Confusion Matrix')
disp(CM)