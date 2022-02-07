%% Example of NClass SVDD

% This script shows an example of N class classification performed via
% multiclass-SVDD.
%
% There are two new main self-signing functions presented in this work:
% NC_SVDD_TRAINING which trains a multiclass SVDD and 
% NC_SVDD_TEST which tests the trained SVDD. 
%
% Other new functions, MixGauss, TestObject_N, KernelMatrix, 
%are necessary for the operation of the previous ones.
%
% Having defined the number of classes, Num_class, 
%the set of observations Xi and the respective target Yi,
% the program performs an SVDD classification. 
%
% The example given here is two-dimensional and with 5 classes,
% but can be extended to an arbitrary number of classes and dimensions.

clc; clear all; close all; %#ok<CLALL>

Num_class = 6;
n = 100;

% X1 = MixGauss([-5;-5],[1,1],n); 
% X2 = MixGauss([5;5],[1,1],n); 
% X3 = MixGauss([1;1],[1,1],n); 
% X4 = MixGauss([-1;-1],[1,1],n); 
% X5 = MixGauss([-1;5],[1,1],n); 

X1 = MixGauss([-5;0],[5,5],n); 
X2 = MixGauss([10;10],[2,2],n); 
X3 = MixGauss([-10;10],[2,2],n); 
X4 = MixGauss([10;-10],[2,2],n); 
X5 = MixGauss([-10;-10],[2,2],n); 
X6 = MixGauss([5;0],[2,2],n);

Xtr = [X1;X2;X3;X4;X5;X6];

Y1 = 1*ones(n,1);
Y2 = 2*ones(n,1);
Y3 = 3*ones(n,1);
Y4 = 4*ones(n,1);
Y5 = 5*ones(n,1);
Y6 = 6*ones(n,1);

Ytr = [Y1;Y2;Y3;Y4;Y5;Y6];

% C1=1; C2=0.5; C3=0.5; C4 = 0.5;
% C5 = 0.5; C6 = 1; C7 = 0.5;
% C8 = 0.5; C9 = 0.5; C10 = 0.5;
% C11=1; C12=0.5; C13=0.5; C14=0.5; C15=0.5; C16=1;
% C17=0.5; C18=0.5; C19=0.5; C20=0.5; C21=1; C22=0.5; C23=0.5; C24=0.5; C25=0.5;
% 
% C = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16 ...
%     C17, C18, C19, C20, C21, C22, C23, C24, C25];

C = ones(1,36);

kernel = 'gaussian'; 
param = 5;

[x_class, Ytr_class, Rsquared_class, a_class, SV_class, YSV_class]=...
    NC_SVDD_TRAINING(Xtr, Ytr, Num_class, kernel, param, C);

x1 = x_class{1};
x2 = x_class{2};
x3 = x_class{3};
x4 = x_class{4};
x5 = x_class{5};
x6 = x_class{6};

Rsquared1 = Rsquared_class{1};
Rsquared2 = Rsquared_class{2};
Rsquared3 = Rsquared_class{3};
Rsquared4 = Rsquared_class{4};
Rsquared5 = Rsquared_class{5};
Rsquared6 = Rsquared_class{6};

Ytr1 = Ytr_class{1};
Ytr2 = Ytr_class{2};
Ytr3 = Ytr_class{3};
Ytr4 = Ytr_class{4};
Ytr5 = Ytr_class{5};
Ytr6 = Ytr_class{6};

SV1 = SV_class{1};
SV2 = SV_class{2};
SV3 = SV_class{3};
SV4 = SV_class{4};
SV5 = SV_class{5};
SV6 = SV_class{6};

%%%%

y_pred = ...
    NC_SVDD_TEST(Xtr, Ytr_class, Num_class, x_class, Xtr, kernel, param, Rsquared_class);

%%%%

dimGrid=100; % dimGrid*dimGrid

[K1, Z1] = meshgrid(linspace(min(Xtr(:,1))-1, max(Xtr(:,1))+1,dimGrid),...
                    linspace(min(Xtr(:,2))-1, max(Xtr(:,2))+1,dimGrid));

x=linspace(min(Xtr(:,1))-1, max(Xtr(:,1))+1, dimGrid);
y=linspace(min(Xtr(:,2))-1, max(Xtr(:,2))+1, dimGrid);
   
K1=K1(:); Z1=Z1(:);
E=[K1 Z1];
    
y_pred2 = ...
    NC_SVDD_TEST(Xtr, Ytr_class, Num_class, x_class, E, kernel, param, Rsquared_class);

%% Plot

figure(1)

TargeT = y_pred2;

TargeT1=TargeT;
TargeT2=TargeT;
TargeT3=TargeT;
TargeT4=TargeT;
TargeT5=TargeT;
TargeT6=TargeT;

TargeT1(TargeT1~=1)=Num_class+1;
TargeT1(TargeT1==1)=-1;

TargeT2(TargeT2~=2)=Num_class+1;
TargeT2(TargeT2==2)=-1;

TargeT3(TargeT3~=3)=Num_class+1;
TargeT3(TargeT3==3)=-1;

TargeT4(TargeT4~=4)=Num_class+1;
TargeT4(TargeT4==4)=-1;

TargeT5(TargeT5~=5)=Num_class+1;
TargeT5(TargeT5==5)=-1;

TargeT6(TargeT6~=6)=Num_class+1;
TargeT6(TargeT6==6)=-1;

c1 = contour(x, y, reshape(TargeT1,numel(y),numel(x)),[0.9999 0.9999] , ...
    'linecolor', 'r', 'LineWidth', 1);
hold on 
c2 = contour(x, y, reshape(TargeT2,numel(y),numel(x)),[0.9999 0.9999] , ...
    'linecolor', 'b', 'LineWidth', 1);
hold on
c3 = contour(x, y, reshape(TargeT3,numel(y),numel(x)),[0.9999 0.9999] , ...
    'linecolor', 'g', 'LineWidth', 1);
hold on
c4 = contour(x, y, reshape(TargeT4,numel(y),numel(x)),[0.9999 0.9999] , ...
    'linecolor', 'k', 'LineWidth', 1);
hold on
c5 = contour(x, y, reshape(TargeT5,numel(y),numel(x)),[0.9999 0.9999] , ...
    'linecolor', 'c', 'LineWidth', 1);
hold on
c6 = contour(x, y, reshape(TargeT6,numel(y),numel(x)),[0.9999 0.9999] , ...
    'linecolor', 'y', 'LineWidth', 1);

% Support Vectors

hold on
p=plot(SV1(:,1),SV1(:,2),'kO','MarkerSize',8,'DisplayName','SV');
hold on
plot(SV2(:,1),SV2(:,2),'kO','MarkerSize',8)
hold on
plot(SV3(:,1),SV3(:,2),'kO','MarkerSize',8)
hold on
plot(SV4(:,1),SV4(:,2),'kO','MarkerSize',8)
hold on
plot(SV5(:,1),SV5(:,2),'kO','MarkerSize',8)
hold on
plot(SV6(:,1),SV6(:,2),'kO','MarkerSize',8)

hold on

g = gscatter(Xtr(:,1), Xtr(:,2), Ytr,'rbgkcy.');

for i = 1:numel(g)
    g(i).DisplayName = strcat(g(i).DisplayName, ', n= ', string(numel(g(i).XData)));
    
end

legend([p g(1) g(2) g(3) g(4) g(5) g(6)])

%% Confusion Matrix

ConfusionMatrix(Ytr, y_pred, Num_class)

figure(2)

cm = confusionchart(Ytr, y_pred);
