function K = CohenKappa(X, Y, y_pred, Num_class)

% Function that computes the CohenKappa index for multiclassification
% algorithms

    CM = ConfusionMatrix(Y, y_pred, Num_class);

    c = sum(diag(CM));

    s = size(X,1);

    p = []; t = [];

    for k =1:Num_class

        p_k = sum(CM(:,k));

        p = [p,p_k];

        t_k = sum(CM(k,:));

        t = [t,t_k];

    end

    K = ((c*s) - sum(p.*t))/(s^2 - sum(p.*t));