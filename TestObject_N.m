function T=TestObject_N(Xtr, Ytr, alpha, Z, kernel, param)

% Function which computes the distance of an observation from the center of
% an SVDD class.

% Xtr: training set
% Ytr: array with the class targets. It must be 
% [1 1 ... 1 2 2 ... 2 ... n n ... n]
% alpha: array of the lagrange multipliers computed by NC_SVDD_TRAINING
% Z: object to test
% kernel: kernel function (linear, polynomial, gaussian)
% param: kernel parameter

alf_i=alpha(Ytr==+1,1);
alf_l=alpha(Ytr==-1,1);

flag_i=find(all(Ytr==+1,2));
flag_l=find(all(Ytr==-1,2));

X_i=Xtr(flag_i,:);
X_l=Xtr(flag_l,:);

 K_i=KernelMatrix(X_i, X_i, kernel, param);
 K_l=KernelMatrix(X_l, X_l, kernel, param);

 Zker=KernelMatrix(Z, Z, kernel, param);
 Kz=diag(Zker);
 
 KZX_i=KernelMatrix(Z,X_i,kernel,param);
 KZX_l=KernelMatrix(Z,X_l,kernel,param);

 KX_lX_i=KernelMatrix(X_l, X_i, kernel, param);
 
 T=Kz-2*(KZX_i*alf_i-KZX_l*alf_l)+ ...
     +alf_i'*K_i*alf_i-2*alf_l'*KX_lX_i*alf_i+alf_l'*K_l*alf_l;
 
end