function D = SquareDist(X1, X2)
    n = size(X1,1);
    m = size(X2,1);

    sq1 = sum(X1.*X1,2);
    sq2 = sum(X2.*X2,2);
    
    D = sq1*ones(1,m) + ones(n,1)*sq2' - 2*(X1*X2');
end
