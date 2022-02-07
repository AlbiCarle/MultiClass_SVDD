function [x_class, Ytr_class, Rsquared_class, a_class, SV_class, YSV_class]=...
    NC_SVDD_TRAINING(Xtr, Ytr, Num_class, kernel, param, C)

% Function which trains a multiclass-SVDD:
% Xtr: training set
% Ytr: array with the class targets. It must be 
% [1 1 ... 1 2 2 ... 2 ... n n ... n]
% Num_class: number of classes
% kernel: kernel function (linear, polynomial, gaussian)
% param: kernel parameter
% C vector of the weights of each pair of classes

N_class = cell(1,Num_class);

for i = 1:Num_class
    N_class{i} = size(Ytr(Ytr==i),1);
end

N = 0;

for i = 1:Num_class
    N = N + N_class{i};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (isequal(kernel,'linear') || isequal(kernel,'polynomial'))

    Ztr = Xtr+10; 
    Ztr = normalize(Ztr, 2,'norm',2);

else

    Ztr = Xtr;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                          % L=-(1/2x'Hx+f'x)
%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K = KernelMatrix(Ztr, Ztr, kernel, param);

%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ytr_class = {};

Ytr_class{1} = [ones(N_class{1},1);-ones(N-N_class{1},1)];

sum1 = 0;

for i = 2:Num_class
    
    sum1 = sum1 + N_class{i-1};

    Ytr_class{i} = -ones(N,1);
    Ytr_class{i}(sum1+1:sum1+N_class{i},1) = ...
        -Ytr_class{i}(sum1+1:sum1+N_class{i},1);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

H_class = {};
f_class = {};

for i=1:Num_class
    
    Hi = Ytr_class{i}*Ytr_class{i}'.*K;
    Hi = Hi+Hi';
    fi = Ytr_class{i}.*diag(K);

    H_class{i} = Hi;
    f_class{i} = fi;
    
end

H = []; f = [];

for i = 1 : Num_class

    H = blkdiag(H_class{i},H);

    f = [f; f_class{i}];

end

%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lb = zeros(Num_class*N,1);

ub = ones(Num_class*N,1);

Ytr_ub = Ytr;

for i =1:Num_class-1
    Ytr_ub =  [Ytr_ub; Ytr+Num_class*i];
end

for i = 1:length(C)
    ub(Ytr_ub==i)=C(i);
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Aeq_class={};

for i=1:Num_class

    Aeq_class{i} = zeros(1,Num_class*N);
    Aeq_class{i}(1,Ytr_class{i}==+1)=+1;
    Aeq_class{i}(1,Ytr_class{i}==-1)=-1;

    Aeq_class{i} = circshift(Aeq_class{i},N*(i-1));
end

Aeq = [];

for i=1:Num_class

    Aeq = [Aeq; Aeq_class{i}];

end

beq=ones(Num_class,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

options = optimset('Display', 'on');

x = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options); %#ok<NASGU> 

%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_class = {}; % alpha^{hk}

for i =1:Num_class

    x_class{i} = x(N*(i-1)+1:i*N,:);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a_class = {}; % list of centers

for i = 1:Num_class

    a_class{i} = x_class{i}'*(Ytr_class{i}.*Xtr);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inc=1E-5;

%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idxSV_class = {};
SV_class = {};
YSV_class = {};

for i = 1:Num_class

    idxSV_class{i} = find(all(abs(x_class{i})>inc & abs(x_class{i})<C(Num_class*(i-1)+i)-inc,2));
    SV_class{i} = Xtr(idxSV_class{i},:); 
    YSV_class{i} = Ytr_class{i}(idxSV_class{i},:);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%% XXXXXXXXXXXXXX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Rsquared_class = {};

for i=1:Num_class

   
    if(size(SV_class{i},1)>0)

        rand=randperm(size(SV_class{i},1),1); 
        x_s=SV_class{i}(rand,:);
        Rsquared_class{i} = TestObject_N(Xtr, Ytr_class{i}, x_class{i}, x_s, kernel, param);
   
    else

        Rsquared_class{i} = 0;

    end

end

%%%%%%%%%%%%%%%%%%