function [param_star, C_star, err_matrix] = ...
    CV_MultiSVDD(X, Y, kernel,nrip, KerPar,Cpar)

% Cross Validation function for MC-SVDD

    Num_class = length(unique(Y));
    minimum_abs = 100;

    for rip = 1:nrip

        disp(['--->', num2str(rip)])

        cv = cvpartition(Y,'HoldOut',0.3, 'Stratify',true);
        idx = cv.test;

        Xtr = X(~idx,:); Ytr = Y(~idx,:);
        Xvl = X(idx,:); Yvl = Y(idx,:);

        err_matrix = zeros(size(KerPar,2),size(Cpar,1));

        i = 0; 
        
        for param = KerPar

            i = i + 1;
            
            j = 0;

            for C = Cpar'

                C = C';

                j = j + 1;

                [x_class, Ytr_class, Rsquared_class, a_class, SV_class, YSV_class]=...
            NC_SVDD_TRAINING(Xtr, Ytr, Num_class, kernel, param, C);

                y_predict = ...
                    NC_SVDD_TEST(Xtr, Ytr_class, Num_class, x_class, Xvl, kernel, param, Rsquared_class);
                
                n = size(Yvl,1);

                err = (n-sum(Yvl == y_predict))/n;

                err_matrix(i,j) = err;

            end
        end

        minimum = min(min(err_matrix));

        if minimum < minimum_abs

            [x,y]=find(err_matrix==minimum);

            param_star = KerPar(x(1));

            C_star = Cpar(y(1),:);

            minimum_abs = minimum;

            disp(minimum_abs)

        end

    end

disp('Done')

end
