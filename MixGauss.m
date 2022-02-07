function [X, Y] = MixGauss(means, sigmas, n)

% EXAMPLE: [X, Y] = MixGauss([[0;0],[1;1]],[0.5,0.25],1000); 
% generates a 2D dataset with two classes, the first one centered on (0,0)
% with standard deviation 0.5, the second one centered on (1,1) with standard deviation 0.25. 
% each class will contain 1000 points
%
% to visualize: scatter(X(:,1),X(:,2),25,Y)

d = size(means,1);
p = size(means,2);

X = [];
Y = [];
for i = 1:p
    m = means(:,i);
    S = sigmas(i);
    Xi = zeros(n,d);
    Yi = zeros(n,1);
    for j = 1:n
        x = S*randn(d,1) + m;
        Xi(j,:) = x;
        Yi(j) = i-1;
    end
    X = [X; Xi];
    Y = [Y; Yi];
end

        
