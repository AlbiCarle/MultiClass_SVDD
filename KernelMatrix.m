function K = KernelMatrix(X1, X2, kernel, param)

    if numel(kernel) == 0
        kernel = 'linear';
    end
    if isequal(kernel, 'linear')
        K = X1*X2';
    elseif isequal(kernel, 'polynomial')
        K = (1 + X1*X2').^param;
    elseif isequal(kernel, 'gaussian')
        K = exp(-1/(2*param^2)*SquareDist(X1,X2));
    end
end
